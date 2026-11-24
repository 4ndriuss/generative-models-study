"""Evaluasi kuantitatif keenam model pada Test Set -> tabel Bagian IV.

Output (outputs/results/):
    metrics.json         semua angka mentah
    metrics_table.md     tabel siap salin ke laporan
    samples_<model>.pt   64 sampel pertama (dipakai ulang oleh make_figures.py)
    samples_<model>.png  grid 8x8 sampel
"""

import torch
from torchvision.utils import save_image

from src import config
from src.cli import common_parser, setup
from src.data import get_splits, subset_tensors
from src.metrics import (extract_features, fid_from_features, measure_latency, mode_coverage,
                         reconstruction_metrics)
from src.models import DISPLAY_NAMES
from src.training import checkpoint_path, load_history, load_model
from src.utils import output_dir, save_json, set_seed


@torch.no_grad()
def generate(sample_fn, n: int, batch: int = 500) -> torch.Tensor:
    out = []
    for i in range(0, n, batch):
        out.append(sample_fn(min(batch, n - i)).cpu())
    return torch.cat(out)


@torch.no_grad()
def gaussian_fit_sampler(model, x_train, device):
    """Untuk AE deterministik (tanpa prior): fit N(mu, Sigma) pada laten data train, lalu sampling.

    Ini BUKAN bagian template (template menulis N/A), tapi berguna untuk menunjukkan secara
    kuantitatif bahwa ruang laten AE tidak terisi rapat (banyak sampel jatuh di latent voids).
    """
    z = torch.cat([model.encode(x_train[i:i + 1000].to(device)).cpu() for i in range(0, len(x_train), 1000)])
    mu = z.mean(0)
    cov = torch.cov(z.T) + 1e-4 * torch.eye(z.shape[1])
    dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    return lambda n: model.decode(dist.sample((n,)).to(device))


@torch.no_grad()
def sparsity_stats(model, x, device) -> dict:
    h = torch.cat([model.hidden(x[i:i + 1000].to(device)).cpu() for i in range(0, len(x), 1000)])
    return dict(frac_inactive=(h <= 1e-3).float().mean().item(),
                active_units_per_image=(h > 1e-3).float().sum(1).mean().item(),
                mean_abs_activation=h.abs().mean().item())


def stability_summary(name: str, hist: dict, coverage: dict | None) -> dict:
    """Ringkasan numerik kestabilan dari kurva loss (25% iterasi terakhir)."""
    it = hist["iter"]
    key = "g_loss" if name in ("gan", "dcgan") else "loss"
    loss = torch.tensor(it[key])
    tail = loss[int(0.75 * len(loss)):] if len(loss) >= 4 else loss
    out = {"tail_loss_mean": tail.mean().item(),
           "tail_loss_cv": (tail.std() / tail.mean().abs().clamp_min(1e-8)).item() if len(tail) > 1 else 0.0}
    if name in ("gan", "dcgan"):
        ep = hist["epoch"]
        out.update(d_real_final=ep["d_real"][-1], d_fake_final=ep["d_fake"][-1],
                   d_loss_final=ep["d_loss"][-1], g_grad_norm_final=ep["g_grad_norm"][-1])
        notes = []
        if out["d_real_final"] > 0.9 and out["d_fake_final"] < 0.1:
            notes.append("D mendominasi (risiko vanishing gradient)")
        if coverage and coverage["norm_entropy"] < 0.85:
            notes.append("indikasi mode collapse")
        notes.append(f"osilasi G (CV)={out['tail_loss_cv']:.2f}")
        out["note"] = "; ".join(notes)
    else:
        ep_loss = hist["epoch"]["loss"]
        n_up = sum(b > a for a, b in zip(ep_loss, ep_loss[1:]))     # berapa kali loss epoch NAIK
        trend = "turun monoton" if n_up == 0 else f"naik di {n_up} dari {len(ep_loss) - 1} transisi epoch"
        out["epochs_with_increase"] = n_up
        out["note"] = f"loss {ep_loss[0]:.3g}->{ep_loss[-1]:.3g} ({trend}), CV ekor={out['tail_loss_cv']:.2f}"
    return out


def fmt(v, nd=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def write_table(results: dict, path) -> None:
    header = ("| Arsitektur Model | Dim Laten | Train Time / Epoch (s) | Latency Generate (ms / batch 64) "
              "| Latency Rekonstruksi (ms / batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID | "
              "FID* (Gaussian-fit) | Mode Coverage (entropi) | Kestabilan Training |")
    sep = "|" + "---|" * 12
    rows = [header, sep]
    for name, r in results.items():
        rec = r.get("reconstruction") or {}
        rows.append("| " + " | ".join([
            DISPLAY_NAMES[name], str(r["latent_dim"]), fmt(r["train_time_per_epoch_s"], 2),
            fmt(r["latency_generate_ms"], 2), fmt(r.get("latency_recon_ms"), 2),
            fmt(rec.get("mse"), 5), fmt(rec.get("psnr"), 2), fmt(rec.get("ssim"), 4),
            fmt(r.get("fid"), 2) if r.get("fid") is not None else "N/A (latent voids)" if name in ("ae", "sae") else "N/A",
            fmt(r.get("fid_gaussian_fit"), 2), fmt((r.get("mode_coverage") or {}).get("norm_entropy"), 3),
            r["stability"]["note"],
        ]) + " |")
    notes = [
        "",
        "Catatan:",
        "- FID dihitung dengan fitur 128-d dari FIDNet (CNN yang dilatih pada data train), "
        f"{results[next(iter(results))]['fid_samples']} sampel sintetis vs seluruh Test Set. Hanya sebanding antarmodel di eksperimen ini.",
        "- FID* untuk Vanilla/Sparse AE: sampling z ~ N(mu, Sigma) yang di-fit pada laten train (bukan prior bawaan model).",
        "- Latency Generate: AE/SAE/VAE = decode 64 vektor laten; GAN/DCGAN = 1 forward G; DDPM = T langkah reverse denoising.",
        "- Mode Coverage = entropi histogram kelas prediksi FIDNet / log(10); 1.0 = semua kelas merata.",
    ]
    path.write_text("\n".join(rows + notes), encoding="utf-8")


def main():
    p = common_parser("Evaluasi metrik Test Set")
    p.add_argument("--fid-samples", type=int, default=config.FID_SAMPLES)
    p.add_argument("--models", nargs="*", default=config.GENERATIVE_MODELS)
    args = p.parse_args()
    device = setup(args)
    n_fid = config.QUICK["fid_samples"] if args.quick else args.fid_samples
    res_dir = output_dir("results")

    train, _, test = get_splits(args.dataset, args.fake_data)
    x_train, _ = subset_tensors(train)
    x_test, _ = subset_tensors(test)
    x_lat = x_test[:config.LATENCY_BATCH].to(device)

    fidnet = load_model("fidnet", device)
    real_feats = extract_features(fidnet, x_test, device)
    print(f"FIDNet siap. Fitur real test: {tuple(real_feats.shape)}")

    results = {}
    for name in args.models:
        if not checkpoint_path(name).exists():
            print(f"[skip] {name}: checkpoint belum ada")
            continue
        print(f"== Evaluasi {name}")
        set_seed(config.SEED)
        model = load_model(name, device)
        hist = load_history(name)
        r = {"latent_dim": "28x28 (dimensi citra)" if name == "ddpm" else model.latent_dim,
             "train_time_per_epoch_s": sum(hist["epoch_time_s"]) / len(hist["epoch_time_s"]),
             "epochs": len(hist["epoch_time_s"]), "n_params": hist.get("n_params"),
             "fid_samples": n_fid}

        if name in ("ae", "sae", "vae"):
            r["reconstruction"] = reconstruction_metrics(model.reconstruct, x_test, device)
            r["latency_recon_ms"] = measure_latency(lambda: model.reconstruct(x_lat), device)["ms"]
        if name in ("ae", "sae"):
            r["sparsity"] = sparsity_stats(model, x_test, device)
            sampler = gaussian_fit_sampler(model, x_train, device)
        else:
            sampler = lambda n: model.sample(n, device)

        runs = dict(n_warmup=1, n_runs=3) if name == "ddpm" else {}
        r["latency_generate_ms"] = measure_latency(lambda: sampler(config.LATENCY_BATCH), device, **runs)["ms"]

        set_seed(config.SEED)
        samples = generate(sampler, n_fid, batch=250 if name == "ddpm" else 500)
        fake_feats = extract_features(fidnet, samples, device)
        fid = fid_from_features(real_feats, fake_feats)
        r["fid_gaussian_fit" if name in ("ae", "sae") else "fid"] = fid
        r["mode_coverage"] = mode_coverage(fidnet, samples, device)
        r["stability"] = stability_summary(name, hist, r["mode_coverage"] if name in ("gan", "dcgan") else None)

        torch.save(samples[:64], res_dir / f"samples_{name}.pt")
        save_image(samples[:64], res_dir / f"samples_{name}.png", nrow=8)
        results[name] = r
        print({k: v for k, v in r.items() if k not in ("stability",)})

    save_json(results, res_dir / "metrics.json")
    if results:
        write_table(results, res_dir / "metrics_table.md")
        print((res_dir / "metrics_table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
