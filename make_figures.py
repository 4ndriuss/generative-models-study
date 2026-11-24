"""Figur wajib Bagian V (Figur 3 dibuat oleh mlp_experiment.py) + figur tambahan untuk analisis.

Output (outputs/figures/):
    fig1_reconstruction.png       GT vs Vanilla AE vs Sparse AE vs VAE
    fig2_random_sampling.png      4 panel 4x4: VAE, Vanilla GAN, DCGAN, DDPM
    fig4_interpolation.png        interpolasi linear laten, alpha 0..1
    fig5a_loss_curves.png         kurva loss keenam model
    fig5b_ddpm_trajectory.png     x_T -> x_0 langkah demi langkah
    fig5c_gan_diagnostics.png     [tambahan] D(x), D(G(z)), norma gradien G
    fig_extra_sae_features.png    [tambahan] filter encoder & sparsity aktivasi AE vs SAE
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src import config
from src.cli import common_parser, setup
from src.data import CLASS_NAMES, get_splits, subset_tensors
from src.models import DISPLAY_NAMES
from src.plotting import grid_image, row_figure, save, show_grid
from src.training import checkpoint_path, load_history, load_model
from src.utils import load_json, output_dir, set_seed


def one_per_class(y, classes):
    return torch.stack([(y == c).nonzero()[0, 0] for c in classes])


def available(names):
    return [n for n in names if checkpoint_path(n).exists()]


# ------------------------------------------------------------------ Figur 1
@torch.no_grad()
def fig1(x_test, y_test, names, device, out):
    idx = one_per_class(y_test, range(8))
    x = x_test[idx].to(device)
    rows = [("Ground Truth", x.cpu())]
    metrics = _metrics()
    for n in available(["ae", "sae", "vae"]):
        m = load_model(n, device)
        label = DISPLAY_NAMES[n].replace(" (VAE)", "")
        if n in metrics and "reconstruction" in metrics[n]:
            r = metrics[n]["reconstruction"]
            label += f"\nMSE={r['mse']:.4f} SSIM={r['ssim']:.3f}"
        rows.append((label, m.reconstruct(x).cpu()))
    row_figure(rows, "Figur 1. Rekonstruksi Test Set: Autoencoder deterministik vs VAE",
               out / "fig1_reconstruction.png", col_labels=[names[i] for i in range(8)])


# ------------------------------------------------------------------ Figur 2
@torch.no_grad()
def _samples(name, device, n=16):
    path = output_dir("results") / f"samples_{name}.pt"
    if path.exists():
        return torch.load(path)[:n]
    set_seed(config.SEED)
    return load_model(name, device).sample(n, device).cpu()


def _metrics():
    path = output_dir("results") / "metrics.json"
    return load_json(path) if path.exists() else {}


def fig2(device, out):
    metrics = _metrics()
    panels = [("A", "vae", "z ~ N(0, I)"), ("B", "gan", "z ~ N(0, I)"),
              ("C", "dcgan", "z ~ N(0, I)"), ("D", "ddpm", "reverse denoising dari x_T ~ N(0, I)")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.6))
    for ax, (letter, name, src) in zip(axes.flat, panels):
        if not checkpoint_path(name).exists():
            ax.axis("off")
            ax.set_title(f"Panel {letter}: {DISPLAY_NAMES[name]} (belum dilatih)")
            continue
        title = f"Panel {letter}: {DISPLAY_NAMES[name]}\n{src}"
        if name in metrics and metrics[name].get("fid") is not None:
            title += f"\nFID={metrics[name]['fid']:.2f}, coverage={metrics[name]['mode_coverage']['norm_entropy']:.2f}"
        show_grid(ax, _samples(name, device), nrow=4, title=title, fontsize=9)
    fig.suptitle("Figur 2. Random generative sampling (4x4 per model)", fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig2_random_sampling.png")


# ------------------------------------------------------------------ Figur 4
@torch.no_grad()
def _gan_endpoints(gan, fidnet, device, cls_a, cls_b, n=2000):
    """GAN tidak punya encoder: cari z yang hasil generate-nya paling yakin diklasifikasikan
    sebagai kelas A dan kelas B oleh FIDNet."""
    set_seed(config.SEED)
    z = torch.randn(n, gan.z_dim, device=device)
    probs = F.softmax(fidnet(gan.decode(z)), 1)
    return z[probs[:, cls_a].argmax()], z[probs[:, cls_b].argmax()]


@torch.no_grad()
def fig4(x_test, y_test, names, device, out):
    a, b = config.INTERP_CLASS_A, config.INTERP_CLASS_B
    ia, ib = one_per_class(y_test, [a, b])
    xa, xb = x_test[ia:ia + 1].to(device), x_test[ib:ib + 1].to(device)
    alphas = torch.tensor(config.ALPHAS, device=device)[:, None]
    rows = [("Ruang piksel\n(baseline)", ((1 - alphas.view(-1, 1, 1, 1)) * xa + alphas.view(-1, 1, 1, 1) * xb).cpu())]
    for n in available(["ae", "sae", "vae"]):
        m = load_model(n, device)
        za, zb = m.encode(xa)[0], m.encode(xb)[0]
        rows.append((DISPLAY_NAMES[n].replace(" (VAE)", ""), m.decode((1 - alphas) * za + alphas * zb).cpu()))
    gans = available(["gan", "dcgan"])
    if gans and checkpoint_path("fidnet").exists():
        fidnet = load_model("fidnet", device)
        for n in gans:
            m = load_model(n, device)
            za, zb = _gan_endpoints(m, fidnet, device, a, b)
            rows.append((DISPLAY_NAMES[n], m.decode((1 - alphas) * za + alphas * zb).cpu()))
    row_figure(rows, f"Figur 4. Interpolasi linear z(alpha) = (1-alpha) z_A + alpha z_B  "
                     f"({names[a]} -> {names[b]})",
               out / "fig4_interpolation.png", col_labels=[f"alpha={v:.1f}" for v in config.ALPHAS])


# ------------------------------------------------------------------ Figur 5
def _plot_curve(ax, hist, key, label, color, raw=True):
    epochs = len(hist["epoch_time_s"])
    it = torch.tensor(hist["iter"][key])
    if raw and len(it) > 0:
        ax.plot(torch.arange(len(it)) / (len(it) / epochs), it, color=color, alpha=0.15, lw=0.6)
    ax.plot(range(1, epochs + 1), hist["epoch"][key], color=color, marker="o", ms=3, lw=1.6, label=label)


def fig5a(out):
    specs = [
        ("ae", [("recon", "MSE rekonstruksi (train)")]),
        ("sae", [("recon", "MSE rekonstruksi"), ("loss", "total = MSE + lambda*L1")]),
        ("vae", [("recon", "BCE rekonstruksi"), ("kl", "KL(q(z|x) || N(0,I))")]),
        ("gan", [("d_loss", "Loss D"), ("g_loss", "Loss G")]),
        ("dcgan", [("d_loss", "Loss D"), ("g_loss", "Loss G")]),
        ("ddpm", [("loss", "MSE prediksi derau")]),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, keys) in zip(axes.flat, specs):
        ax.set_title(DISPLAY_NAMES[name], fontsize=11)
        if not checkpoint_path(name).exists():
            ax.text(0.5, 0.5, "belum dilatih", ha="center", transform=ax.transAxes)
            continue
        hist = load_history(name)
        for (key, label), c in zip(keys, colors):
            target = ax.twinx() if (name == "vae" and key == "kl") else ax
            _plot_curve(target, hist, key, label, c)
            if target is not ax:
                target.set_ylabel(label, color=c)
                target.legend(loc="center right", fontsize=8)
        if "val" in hist and "loss" in hist["val"] and name != "vae":
            ax.plot(range(1, len(hist["val"]["loss"]) + 1), hist["val"]["loss"], "k--", lw=1.2, label="validasi (loss)")
        if name == "ddpm":
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Figur 5(a). Kurva training loss (garis tipis = per iterasi, titik = rata-rata epoch)",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig5a_loss_curves.png")


def fig5c(out):
    names = available(["gan", "dcgan"])
    if not names:
        return
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 3.6 * len(names)), squeeze=False)
    for row, name in zip(axes, names):
        hist = load_history(name)
        _plot_curve(row[0], hist, "d_real", "D(x) real", "tab:green")
        _plot_curve(row[0], hist, "d_fake", "D(G(z)) fake", "tab:red")
        row[0].axhline(0.5, color="gray", ls=":", lw=1)
        row[0].set_title(f"{DISPLAY_NAMES[name]}: output discriminator (ideal Nash -> 0.5)")
        _plot_curve(row[1], hist, "g_grad_norm", "||grad G||", "tab:purple")
        row[1].set_yscale("log")
        row[1].set_title(f"{DISPLAY_NAMES[name]}: norma gradien generator")
        for ax in row:
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("Figur 5(c) [tambahan]. Diagnostik stabilitas GAN", fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig5c_gan_diagnostics.png")


@torch.no_grad()
def fig5b(device, out, n=6):
    if not checkpoint_path("ddpm").exists():
        return
    ddpm = load_model("ddpm", device)
    T = ddpm.T
    steps = sorted({T, int(0.9 * T), int(0.75 * T), int(0.5 * T), int(0.25 * T),
                    int(0.1 * T), int(0.05 * T), 0}, reverse=True)
    set_seed(config.SEED)
    _, traj = ddpm.sample(n, device, trajectory_steps=steps)
    fig, axes = plt.subplots(1, len(steps), figsize=(1.6 * len(steps), 1.6 * n * 0.62 + 0.8))
    for ax, t in zip(axes, steps):
        ax.imshow(grid_image(traj[t], nrow=1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t = {t}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Figur 5(b). Trajektori reverse diffusion x_T -> x_0 (T = {T})", fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig5b_ddpm_trajectory.png")


# ------------------------------------------------------------- tambahan Q2
@torch.no_grad()
def fig_extra_sae(x_test, device, out):
    names = available(["ae", "sae"])
    if not names:
        return
    fig, axes = plt.subplots(2, len(names), figsize=(6 * len(names), 9), squeeze=False)
    for col, name in enumerate(names):
        m = load_model(name, device)
        W = m.enc_hidden[1].weight[:64].cpu().view(-1, 1, 28, 28)       # 64 filter pertama W1
        flat = W.flatten(1)
        W = ((flat - flat.min(1, keepdim=True).values) /
             (flat.max(1, keepdim=True).values - flat.min(1, keepdim=True).values).clamp_min(1e-8)).view_as(W)
        show_grid(axes[0, col], W, nrow=8, title=f"{DISPLAY_NAMES[name]}: 64 filter encoder (baris W1)")
        h = m.hidden(x_test[:2000].to(device)).cpu().flatten()
        frac = (h <= 1e-3).float().mean().item()
        axes[1, col].hist(h.numpy(), bins=80, log=True, color="tab:blue" if name == "ae" else "tab:orange")
        axes[1, col].set_title(f"Distribusi aktivasi hidden h (256-d): {frac:.1%} unit ~ 0")
        axes[1, col].set_xlabel("nilai aktivasi")
    fig.suptitle("Figur tambahan. Struktur fitur Vanilla AE vs Sparse AE", fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig_extra_sae_features.png")


def main():
    p = common_parser("Buat figur laporan")
    p.add_argument("--figs", nargs="*", default=["1", "2", "4", "5", "extra"])
    args = p.parse_args()
    device = setup(args)
    out = output_dir("figures")
    names = CLASS_NAMES[args.dataset]
    _, _, test = get_splits(args.dataset, args.fake_data)
    x_test, y_test = subset_tensors(test)

    if "1" in args.figs:
        fig1(x_test, y_test, names, device, out)
    if "2" in args.figs:
        fig2(device, out)
    if "4" in args.figs:
        fig4(x_test, y_test, names, device, out)
    if "5" in args.figs:
        fig5a(out)
        fig5b(device, out)
        fig5c(out)
    if "extra" in args.figs:
        fig_extra_sae(x_test, device, out)


if __name__ == "__main__":
    main()
