"""Loop training, simpan/muat checkpoint, dan riwayat loss.

Dua jenis loop:
  * train_simple : model dengan satu objektif (AE, SAE, VAE, DDPM, MLP, FIDNet). Tiap model
                   menyediakan method loss(x, y) -> (tensor_loss, dict_log).
  * train_gan    : dua optimizer bergantian (D lalu G) untuk Vanilla GAN dan DCGAN.
"""
import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from . import config
from .data import get_loaders
from .models import build_model
from .utils import Timer, load_json, output_dir, save_json, set_seed


# ---------------------------------------------------------------- checkpoint
def checkpoint_path(name: str):
    return output_dir(name) / "model.pt"


def save_checkpoint(name: str, model, model_kwargs: dict, train_cfg: dict, history: dict) -> None:
    torch.save({"name": name, "model_kwargs": model_kwargs, "train_cfg": train_cfg,
                "state_dict": model.state_dict()}, checkpoint_path(name))
    save_json(history, output_dir(name) / "history.json")


def load_model(name: str, device):
    path = checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint '{name}' belum ada. Jalankan: python train.py --model {name}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(name, **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def load_history(name: str) -> dict:
    return load_json(output_dir(name) / "history.json")


def resolve_config(name: str, epochs=None, lr=None, quick: bool = False, **overrides):
    """Gabungkan config default + override CLI + penyesuaian mode --quick."""
    cfg = copy.deepcopy(config.MODEL_CONFIGS[name])
    kwargs = cfg.pop("kwargs")
    for k, v in overrides.items():
        if v is not None and k in kwargs:
            kwargs[k] = v
    if quick:
        cfg["epochs"] = config.QUICK["epochs"]
        if name == "ddpm":
            kwargs["timesteps"] = config.QUICK["ddpm_timesteps"]
    if epochs is not None:
        cfg["epochs"] = epochs
    if lr is not None:
        cfg["lr"] = lr
    return kwargs, cfg


# --------------------------------------------------------------- loop umum
@torch.no_grad()
def _evaluate_loss(model, loader, device) -> dict:
    model.eval()
    sums, n = defaultdict(float), 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, logs = model.loss(x, y)
        for k, v in logs.items():
            sums[k] += v * x.size(0)
        n += x.size(0)
    return {k: v / n for k, v in sums.items()}


def train_simple(name: str, model, loaders, device, cfg: dict) -> dict:
    train_loader, val_loader, _ = loaders
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    history = {"iter": defaultdict(list), "epoch": defaultdict(list),
               "val": defaultdict(list), "epoch_time_s": []}
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums, n = defaultdict(float), 0
        with Timer(device) as timer:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss, logs = model.loss(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
                if hasattr(model, "update_ema"):
                    model.update_ema()
                for k, v in logs.items():
                    history["iter"][k].append(v)
                    sums[k] += v * x.size(0)
                n += x.size(0)
        history["epoch_time_s"].append(timer.elapsed)
        for k, v in sums.items():
            history["epoch"][k].append(v / n)
        for k, v in _evaluate_loss(model, val_loader, device).items():
            history["val"][k].append(v)
        msg = " ".join(f"{k}={v[-1]:.4f}" for k, v in history["epoch"].items())
        val_msg = " ".join(f"val_{k}={v[-1]:.4f}" for k, v in history["val"].items())
        print(f"[{name}] epoch {epoch}/{cfg['epochs']} {msg} {val_msg} ({timer.elapsed:.1f}s)", flush=True)
    return history


# ---------------------------------------------------------------- loop GAN
def _grad_norm(params) -> float:
    total = sum(p.grad.detach().pow(2).sum() for p in params if p.grad is not None)
    return float(total) ** 0.5


def train_gan(name: str, model, loaders, device, cfg: dict) -> dict:
    """Satu iterasi = 1 langkah D lalu 1 langkah G.

    D memaksimalkan log D(x) + log(1 - D(G(z)))  <=> minimalkan BCE(real,1) + BCE(fake,0)
    G (non-saturating) meminimalkan -log D(G(z)) <=> BCE(D(G(z)), 1)

    Dicatat per iterasi: loss D & G, rata-rata D(x) dan D(G(z)), serta norma gradien G
    (untuk mendeteksi vanishing gradient) -> bahan analisis Q3.
    """
    train_loader = loaders[0]
    betas = cfg.get("betas", (0.5, 0.999))
    opt_d = torch.optim.Adam(model.D.parameters(), lr=cfg["lr"], betas=betas)
    opt_g = torch.optim.Adam(model.G.parameters(), lr=cfg["lr"], betas=betas)
    fixed_z = torch.randn(64, model.z_dim, device=device)     # z tetap untuk memantau progres
    progress_dir = output_dir(name, "progress")
    history = {"iter": defaultdict(list), "epoch": defaultdict(list), "epoch_time_s": []}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        sums, n_batches = defaultdict(float), 0
        with Timer(device) as timer:
            for x, _ in train_loader:
                real = x.to(device) * 2 - 1
                b = real.size(0)
                z = torch.randn(b, model.z_dim, device=device)

                # ---- langkah Discriminator
                fake = model.G(z)
                logit_real, logit_fake = model.D(real), model.D(fake.detach())
                d_loss = (F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real))
                          + F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake)))
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

                # ---- langkah Generator
                logit_gen = model.D(fake)
                if model.g_loss == "saturating":
                    g_loss = -F.softplus(logit_gen).mean()          # = E[log(1 - D(G(z)))]
                else:
                    g_loss = F.binary_cross_entropy_with_logits(logit_gen, torch.ones_like(logit_gen))
                opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                g_grad = _grad_norm(model.G.parameters())
                opt_g.step()

                logs = {"d_loss": d_loss.item(), "g_loss": g_loss.item(),
                        "d_real": torch.sigmoid(logit_real).mean().item(),
                        "d_fake": torch.sigmoid(logit_fake).mean().item(),
                        "g_grad_norm": g_grad}
                for k, v in logs.items():
                    history["iter"][k].append(v)
                    sums[k] += v
                n_batches += 1
        history["epoch_time_s"].append(timer.elapsed)
        for k, v in sums.items():
            history["epoch"][k].append(v / n_batches)
        model.eval()
        with torch.no_grad():
            save_image(model.decode(fixed_z), progress_dir / f"epoch_{epoch:03d}.png", nrow=8)
        e = history["epoch"]
        print(f"[{name}] epoch {epoch}/{cfg['epochs']} d_loss={e['d_loss'][-1]:.3f} "
              f"g_loss={e['g_loss'][-1]:.3f} D(x)={e['d_real'][-1]:.2f} D(G(z))={e['d_fake'][-1]:.2f} "
              f"|grad G|={e['g_grad_norm'][-1]:.3f} ({timer.elapsed:.1f}s)", flush=True)
    return history


# ------------------------------------------------------------- entry point
def train_model(name: str, device, dataset: str = config.DATASET, fake: bool = False,
                quick: bool = False, batch_size: int = config.BATCH_SIZE, epochs=None, lr=None,
                **overrides):
    set_seed(config.SEED)
    kwargs, cfg = resolve_config(name, epochs, lr, quick, **overrides)
    subset = config.QUICK["train_subset"] if quick else None
    loaders = get_loaders(batch_size, dataset, fake, train_subset=subset)
    model = build_model(name, **kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"== Training {name} | params={n_params:,} | kwargs={kwargs} | cfg={cfg}", flush=True)

    trainer = train_gan if name in ("gan", "dcgan") else train_simple
    history = trainer(name, model, loaders, device, cfg)
    history["n_params"] = n_params
    train_cfg = {**cfg, "batch_size": batch_size, "dataset": dataset, "fake_data": fake, "quick": quick}
    save_checkpoint(name, model, kwargs, train_cfg, history)
    print(f"== Saved {checkpoint_path(name)}", flush=True)
    return model.eval()


def ensure_trained(name: str, device, **train_kwargs):
    """Muat checkpoint bila ada; bila belum, latih dulu."""
    if checkpoint_path(name).exists():
        return load_model(name, device)
    return train_model(name, device, **train_kwargs)
