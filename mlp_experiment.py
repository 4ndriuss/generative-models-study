"""Bagian 2: pembuktian empiris bahwa bobot & fitur laten MLP Classifier tidak bisa langsung
dipakai untuk rekonstruksi / image generation.

Tahap 1  Latih MLP 784-256-64-10 (CE), akurasi test >= 88%, simpan W1, W2, W3, ekstrak z_MLP.
Tahap 2  Proyeksi balik dari one-hot y_c:  x = sigmoid(W1^T relu(W2^T relu(W3^T y_c)))
Tahap 3  Proyeksi balik dari z_MLP citra uji: x = sigmoid(W1^T relu(W2^T z_MLP))
Tahap 4  Frozen latent decoding: decoder 64-256-784 (MSE) di atas z_MLP beku vs Vanilla AE.
Tahap 5  Sampling z ~ N(mu_z, diag(sigma_z^2)) di ruang z_MLP -> decoder Tahap 4.

Ditambah diagnostik numerik untuk ketiga batasan teoretis:
  (1) information bottleneck  -> fraksi varians intra-kelas & keragaman rekonstruksi per kelas
  (2) W^T W != I dan ReLU     -> galat ortogonalitas, rank/null space, fraksi pre-aktivasi negatif
  (3) tanpa prior P(z)        -> fraksi z_MLP = 0 vs fraksi sampel Gaussian < 0, FID sampel
Output: outputs/mlp_experiment/ (results.json, summary.md, fig3_mlp_limitations.png)
"""
import torch
import torch.nn.functional as F

from src import config
from src.cli import common_parser, data_kwargs, setup
from src.data import CLASS_NAMES, get_loaders, get_splits, subset_tensors
from src.metrics import (extract_features, fid_from_features, mode_coverage, mse_per_image,
                         paired_metrics)
from src.models import MLPDecoder
from src.plotting import row_figure
from src.training import checkpoint_path, ensure_trained, load_model, resolve_config
from src.utils import Timer, output_dir, save_json, set_seed


# ----------------------------------------------------------------- helpers
@torch.no_grad()
def batched(fn, x, device, batch=1000):
    return torch.cat([fn(x[i:i + batch].to(device)).cpu() for i in range(0, len(x), batch)])


def weight_diagnostics(W: torch.Tensor) -> dict:
    """W berbentuk (out, in). Transpos hanya sama dengan invers bila W ortogonal (W^T W = I).
    Untuk W persegi panjang out < in, rank <= out sehingga W^T W (in x in) PASTI singular:
    ada null space berdimensi >= in - out yang informasinya hilang permanen.
    """
    out_dim, in_dim = W.shape
    s = torch.linalg.svdvals(W)
    rank = int((s > s.max() * 1e-5).sum())
    eye_in, eye_out = torch.eye(in_dim), torch.eye(out_dim)
    return dict(shape=[out_dim, in_dim], rank=rank, null_space_dim=in_dim - rank,
                sigma_max=s.max().item(), sigma_min=s.min().item(),
                orth_err_WtW=(torch.linalg.norm(W.T @ W - eye_in) / torch.linalg.norm(eye_in)).item(),
                orth_err_WWt=(torch.linalg.norm(W @ W.T - eye_out) / torch.linalg.norm(eye_out)).item())


def within_class_fraction(z: torch.Tensor, y: torch.Tensor) -> float:
    """Porsi varians total yang berasal dari variasi DI DALAM kelas (trace S_W / trace S_T).
    Kecil = representasi membuang variasi intra-kelas (gejala information bottleneck CE)."""
    total = ((z - z.mean(0)) ** 2).sum()
    within = sum(((z[y == c] - z[y == c].mean(0)) ** 2).sum() for c in y.unique())
    return (within / total).item()


def intra_class_diversity_ratio(recon: torch.Tensor, orig: torch.Tensor, y: torch.Tensor,
                                per_class: int = 200) -> float:
    """Rata-rata jarak pasangan antar-rekonstruksi dalam satu kelas / jarak antar-asli.
    1.0 = keragaman intra-kelas terjaga; mendekati 0 = semua rekonstruksi jadi 'template' kelas."""
    ratios = []
    for c in y.unique():
        idx = (y == c).nonzero().flatten()[:per_class]
        d_r = torch.pdist(recon[idx].flatten(1)).mean()
        d_o = torch.pdist(orig[idx].flatten(1)).mean()
        ratios.append((d_r / d_o).item())
    return sum(ratios) / len(ratios)


def per_class_mse(recon, orig, y, names):
    m = mse_per_image(recon, orig)
    return {names[int(c)]: m[y == c].mean().item() for c in y.unique()}


def minmax(x: torch.Tensor) -> torch.Tensor:
    """Normalisasi kontras per gambar, HANYA untuk visualisasi."""
    flat = x.flatten(1)
    lo, hi = flat.min(1, keepdim=True).values, flat.max(1, keepdim=True).values
    return ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(x)


def train_frozen_decoder(mlp, device, args, epochs):
    """Tahap 4: hanya decoder yang dilatih; seluruh bobot MLP dibekukan (requires_grad=False)."""
    set_seed(config.SEED)
    mlp.requires_grad_(False).eval()
    subset = config.QUICK["train_subset"] if args.quick else None
    train_loader, val_loader, _ = get_loaders(config.BATCH_SIZE, args.dataset, args.fake_data, subset)
    dec = MLPDecoder(64, 256).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    hist = {"loss": [], "val_loss": [], "epoch_time_s": []}
    for ep in range(1, epochs + 1):
        dec.train()
        tot, n = 0.0, 0
        with Timer(device) as t:
            for x, _ in train_loader:
                x = x.to(device)
                with torch.no_grad():
                    z = mlp.features(x)
                loss = F.mse_loss(dec(z), x, reduction="sum") / x.size(0)   # sama dengan loss Vanilla AE
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * x.size(0)
                n += x.size(0)
        dec.eval()
        with torch.no_grad():
            vl = sum(F.mse_loss(dec(mlp.features(x.to(device))), x.to(device), reduction="sum").item()
                     for x, _ in val_loader) / len(val_loader.dataset)
        hist["loss"].append(tot / n)
        hist["val_loss"].append(vl)
        hist["epoch_time_s"].append(t.elapsed)
        print(f"[frozen-decoder] epoch {ep}/{epochs} loss={tot / n:.3f} val={vl:.3f}", flush=True)
    return dec.eval(), hist


# -------------------------------------------------------------------- main
def main():
    p = common_parser("Eksperimen batasan MLP Classifier (Bagian 2)")
    p.add_argument("--retrain-decoder", action="store_true")
    args = p.parse_args()
    device = setup(args)
    out = output_dir("mlp_experiment")
    names = CLASS_NAMES[args.dataset]
    set_seed(config.SEED)

    train, _, test = get_splits(args.dataset, args.fake_data)
    x_test, y_test = subset_tensors(test)
    R = {}

    # ---------------------------------------------------------- Tahap 1
    mlp = ensure_trained("mlp", device, **data_kwargs(args))
    with torch.no_grad():
        acc = (batched(mlp, x_test, device).argmax(1) == y_test).float().mean().item()
    R["stage1_test_accuracy"] = acc
    print(f"[Tahap 1] akurasi test MLP = {acc:.4f} (syarat >= 0.88: {'OK' if acc >= 0.88 else 'BELUM'})")
    W1, W2, W3 = (m.weight.detach().cpu() for m in (mlp.fc1, mlp.fc2, mlp.fc3))
    b1, b2 = mlp.fc1.bias.detach().cpu(), mlp.fc2.bias.detach().cpu()
    torch.save({"W1": W1, "W2": W2, "W3": W3, "b1": b1, "b2": b2,
                "b3": mlp.fc3.bias.detach().cpu()}, out / "mlp_weights.pt")
    z_mlp = batched(mlp.features, x_test, device)                   # (N, 64)
    torch.save({"z_mlp_test": z_mlp, "y_test": y_test}, out / "z_mlp_test.pt")

    # Diagnostik batasan (2): non-ortogonal & ReLU ireversibel
    R["weights"] = {k: weight_diagnostics(W) for k, W in (("W1", W1), ("W2", W2), ("W3", W3))}
    with torch.no_grad():
        pre1 = x_test.flatten(1) @ W1.T + b1
        pre2 = F.relu(pre1) @ W2.T + b2
    R["relu_zeroed_fraction"] = {"layer1": (pre1 < 0).float().mean().item(),
                                 "layer2": (pre2 < 0).float().mean().item()}

    # ---------------------------------------------------------- Tahap 2
    eye = torch.eye(10)
    with torch.no_grad():
        z_hat = F.relu(eye @ W3)              # W3^T y_c   (baris-vektor: y @ W3 = (W3^T y)^T)
        h_hat = F.relu(z_hat @ W2)            # W2^T z_hat
        x_cls = torch.sigmoid(h_hat @ W1).view(10, 1, 28, 28)
        pred_cls = batched(mlp, x_cls, device).argmax(1)
    class_means = torch.stack([x_test[y_test == c].mean(0) for c in range(10)])
    R["stage2"] = {
        "pixel_range": [x_cls.min().item(), x_cls.max().item()],
        "pixel_std_mean": x_cls.flatten(1).std(1).mean().item(),
        "mse_vs_class_mean_image": mse_per_image(x_cls, class_means).mean().item(),
        "mlp_classifies_projection_as_target": (pred_cls == torch.arange(10)).float().mean().item(),
    }

    # ---------------------------------------------------------- Tahap 3
    with torch.no_grad():
        x_t = torch.sigmoid(F.relu(z_mlp @ W2) @ W1).view(-1, 1, 28, 28)
        # Tambahan: invers linear terbaik (pseudo-inverse Moore-Penrose) + koreksi bias.
        # Bila ini pun gagal, informasinya memang sudah hilang (null space + ReLU), bukan sekadar W^T != W^-1.
        x_p = ((z_mlp - b2) @ torch.linalg.pinv(W2).T - b1) @ torch.linalg.pinv(W1).T
        x_p = x_p.clamp(0, 1).view(-1, 1, 28, 28)
    R["stage3_transpose"] = paired_metrics(x_t, x_test)
    R["stage3_pseudoinverse"] = paired_metrics(x_p, x_test)

    # ---------------------------------------------------------- Tahap 4
    ae = ensure_trained("ae", device, **data_kwargs(args))
    dec_path = out / "frozen_decoder.pt"
    if dec_path.exists() and not args.retrain_decoder:
        dec = MLPDecoder(64, 256).to(device)
        dec.load_state_dict(torch.load(dec_path, map_location=device))
        dec.eval()
    else:
        _, ae_cfg = resolve_config("ae", quick=args.quick)        # epoch sama dengan Vanilla AE
        dec, dec_hist = train_frozen_decoder(mlp, device, args, ae_cfg["epochs"])
        torch.save(dec.state_dict(), dec_path)
        save_json(dec_hist, out / "frozen_decoder_history.json")

    recon_mlp = batched(lambda x: dec(mlp.features(x)), x_test, device)
    recon_ae = batched(ae.reconstruct, x_test, device)
    z_ae = batched(ae.encode, x_test, device)
    R["stage4"] = {
        "frozen_mlp_decoder": paired_metrics(recon_mlp, x_test),
        "vanilla_ae": paired_metrics(recon_ae, x_test),
        "per_class_mse_frozen_mlp": per_class_mse(recon_mlp, x_test, y_test, names),
        "per_class_mse_vanilla_ae": per_class_mse(recon_ae, x_test, y_test, names),
        "intra_class_diversity_ratio_frozen_mlp": intra_class_diversity_ratio(recon_mlp, x_test, y_test),
        "intra_class_diversity_ratio_vanilla_ae": intra_class_diversity_ratio(recon_ae, x_test, y_test),
        "within_class_variance_fraction": {
            "pixels": within_class_fraction(x_test.flatten(1), y_test),
            "z_ae": within_class_fraction(z_ae, y_test),
            "z_mlp": within_class_fraction(z_mlp, y_test),
        },
    }

    # ---------------------------------------------------------- Tahap 5
    set_seed(config.SEED)
    n_samp = config.QUICK["fid_samples"] if args.quick else config.FID_SAMPLES
    mu, sd = z_mlp.mean(0), z_mlp.std(0)
    z_rand = mu + sd * torch.randn(n_samp, 64)
    samples_mlp = batched(dec, z_rand, device)
    mu_ae, sd_ae = z_ae.mean(0), z_ae.std(0)                       # kontrol: prosedur sama pada z_AE
    samples_ae = batched(ae.decode, mu_ae + sd_ae * torch.randn(n_samp, 64), device)
    R["stage5"] = {
        "z_mlp_fraction_exact_zero": (z_mlp == 0).float().mean().item(),
        "z_mlp_dead_dims": int((z_mlp.max(0).values == 0).sum()),
        "z_rand_fraction_negative": (z_rand < 0).float().mean().item(),
    }
    if checkpoint_path("fidnet").exists():
        fidnet = load_model("fidnet", device)
        real = extract_features(fidnet, x_test, device)
        R["stage5"]["fid_mlp_prior_sampling"] = fid_from_features(real, extract_features(fidnet, samples_mlp, device))
        R["stage5"]["fid_ae_same_procedure"] = fid_from_features(real, extract_features(fidnet, samples_ae, device))
        R["stage5"]["mode_coverage_mlp"] = mode_coverage(fidnet, samples_mlp, device)
        if checkpoint_path("vae").exists():
            vae = load_model("vae", device)
            set_seed(config.SEED)
            s_vae = batched(lambda z: vae.decode(z), torch.randn(n_samp, vae.latent_dim), device)
            R["stage5"]["fid_vae_prior_reference"] = fid_from_features(real, extract_features(fidnet, s_vae, device))
    else:
        print("[Tahap 5] FIDNet belum dilatih -> FID dilewati (python train.py --model fidnet)")

    save_json(R, out / "results.json")
    _write_summary(R, out / "summary.md")

    # ---------------------------------------------------------- Figur 3
    idx = torch.stack([(y_test == c).nonzero()[0, 0] for c in range(10)])  # 1 citra per kelas
    rows = [
        ("(1) Citra Asli", x_test[idx]),
        ("(2a) W^T dari one-hot y_c\n(kontras dinormalisasi)", minmax(x_cls)),
        ("(2b) W^T dari z_MLP\n(kontras dinormalisasi)", minmax(x_t[idx])),
        ("(2c) W^+ pseudo-inverse\ndari z_MLP [tambahan]", minmax(x_p[idx])),
        ("(3a) Frozen decoder\ndari z_MLP", recon_mlp[idx]),
        ("(3b) Vanilla AE\ndari z_AE", recon_ae[idx]),
        ("(4) Sampling z~N(mu,sigma^2)\ndi ruang z_MLP", samples_mlp[:10]),
        ("(kontrol) prosedur sama\ndi ruang z_AE", samples_ae[:10]),
    ]
    row_figure(rows, "Figur 3. Bukti empiris batasan bobot & fitur laten MLP Classifier",
               out / "fig3_mlp_limitations.png", col_labels=names)
    print(open(out / "summary.md", encoding="utf-8").read())


def _write_summary(R: dict, path) -> None:
    s4, s5 = R["stage4"], R["stage5"]
    lines = [
        "# Ringkasan Eksperimen Batasan MLP Classifier",
        "",
        f"- Tahap 1 - akurasi test MLP: **{R['stage1_test_accuracy']:.4f}**",
        "- Diagnostik bobot (W berbentuk out x in):",
    ]
    for k, d in R["weights"].items():
        lines.append(f"  - {k} {d['shape']}: rank={d['rank']}, null space={d['null_space_dim']}, "
                     f"||W^T W - I||_F/||I||_F={d['orth_err_WtW']:.3f}, ||W W^T - I||_F/||I||_F={d['orth_err_WWt']:.3f}")
    rz = R["relu_zeroed_fraction"]
    lines += [
        f"- Fraksi pre-aktivasi negatif yang dipotong ReLU: layer1={rz['layer1']:.3f}, layer2={rz['layer2']:.3f}",
        f"- Tahap 2 - rentang piksel proyeksi W^T y_c: {R['stage2']['pixel_range'][0]:.3f}..{R['stage2']['pixel_range'][1]:.3f}, "
        f"MLP mengklasifikasikan proyeksi sebagai kelas target: {R['stage2']['mlp_classifies_projection_as_target']:.0%}",
        "",
        "| Metode rekonstruksi | MSE | PSNR (dB) | SSIM |",
        "|---|---|---|---|",
    ]
    for label, m in [("Tahap 3: W^T dari z_MLP", R["stage3_transpose"]),
                     ("Tahap 3 (tambahan): pseudo-inverse dari z_MLP", R["stage3_pseudoinverse"]),
                     ("Tahap 4: frozen decoder dari z_MLP", s4["frozen_mlp_decoder"]),
                     ("Vanilla AE (z_AE, dim 64)", s4["vanilla_ae"])]:
        lines.append(f"| {label} | {m['mse']:.5f} | {m['psnr']:.2f} | {m['ssim']:.4f} |")
    wc = s4["within_class_variance_fraction"]
    lines += [
        "",
        f"- Rasio keragaman intra-kelas rekonstruksi (1 = terjaga): frozen MLP={s4['intra_class_diversity_ratio_frozen_mlp']:.3f}, "
        f"Vanilla AE={s4['intra_class_diversity_ratio_vanilla_ae']:.3f}",
        f"- Fraksi varians intra-kelas: piksel={wc['pixels']:.3f}, z_AE={wc['z_ae']:.3f}, z_MLP={wc['z_mlp']:.3f}",
        f"- Tahap 5 - z_MLP bernilai tepat 0: {s5['z_mlp_fraction_exact_zero']:.1%} (dimensi mati: {s5['z_mlp_dead_dims']}), "
        f"sedangkan sampel Gaussian bernilai negatif: {s5['z_rand_fraction_negative']:.1%} (wilayah yang TIDAK pernah dilihat decoder)",
    ]
    if "fid_mlp_prior_sampling" in s5:
        lines.append(f"- FID sampel prior z_MLP={s5['fid_mlp_prior_sampling']:.2f}, kontrol z_AE={s5['fid_ae_same_procedure']:.2f}"
                     + (f", referensi VAE prior={s5['fid_vae_prior_reference']:.2f}" if "fid_vae_prior_reference" in s5 else ""))
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
