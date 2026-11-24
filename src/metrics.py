"""Metrik evaluasi: MSE, PSNR, SSIM, FID, mode coverage, dan latensi.

Semua input gambar berada di rentang [0, 1] dengan bentuk (N, 1, 28, 28).
"""
import math
import time

import torch
import torch.nn.functional as F

from .utils import sync


# ---------------------------------------------------------------- rekonstruksi
def mse_per_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).flatten(1).mean(1)


def psnr_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """PSNR = 10 log10(MAX^2 / MSE). Dihitung per gambar lalu dirata-rata."""
    return 10 * torch.log10(data_range ** 2 / mse_per_image(x, y).clamp_min(1e-10))


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]          # (1, 1, size, size)


def ssim_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """SSIM (Wang et al., 2004) dengan jendela Gaussian 11x11, sigma 1.5.

    SSIM membandingkan rata-rata lokal (luminance), varians lokal (contrast), dan
    kovarians lokal (structure) -- lebih peka terhadap hilangnya tepi/tekstur dibanding MSE.
    """
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ch = x.shape[1]
    w = _gaussian_window(device=x.device).repeat(ch, 1, 1, 1)
    mu_x = F.conv2d(x, w, groups=ch)
    mu_y = F.conv2d(y, w, groups=ch)
    sxx = F.conv2d(x * x, w, groups=ch) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=ch) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=ch) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return (num / den).flatten(1).mean(1)


def _summarize(mse, psnr, ssim) -> dict:
    return dict(mse=mse.mean().item(), mse_std=mse.std().item(),
                psnr=psnr.mean().item(), psnr_std=psnr.std().item(),
                ssim=ssim.mean().item(), ssim_std=ssim.std().item())


@torch.no_grad()
def reconstruction_metrics(recon_fn, x: torch.Tensor, device, batch: int = 500) -> dict:
    """recon_fn: fungsi gambar -> rekonstruksi (keduanya [0, 1]). Rata-rata per gambar."""
    mse, psnr, ssim = [], [], []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(device)
        rb = recon_fn(xb).clamp(0, 1)
        mse.append(mse_per_image(rb, xb).cpu())
        psnr.append(psnr_per_image(rb, xb).cpu())
        ssim.append(ssim_per_image(rb, xb).cpu())
    return _summarize(torch.cat(mse), torch.cat(psnr), torch.cat(ssim))


@torch.no_grad()
def paired_metrics(recon: torch.Tensor, orig: torch.Tensor) -> dict:
    """Seperti reconstruction_metrics, tapi rekonstruksinya sudah dihitung (tensor sejajar)."""
    r = recon.clamp(0, 1)
    return _summarize(mse_per_image(r, orig), psnr_per_image(r, orig), ssim_per_image(r, orig))


# ----------------------------------------------------------------------- FID
@torch.no_grad()
def extract_features(fidnet, images: torch.Tensor, device, batch: int = 500) -> torch.Tensor:
    fidnet.eval()
    feats = [fidnet.features(images[i:i + batch].to(device)).double().cpu()
             for i in range(0, len(images), batch)]
    return torch.cat(feats)


def feature_stats(feats: torch.Tensor):
    return feats.mean(0), torch.cov(feats.T)


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """FID = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^{1/2}).

    Tr((S1 S2)^{1/2}) dihitung stabil sebagai jumlah akar eigenvalue dari
    S1^{1/2} S2 S1^{1/2} (matriks simetris semi-definit positif), tanpa scipy.
    """
    diff = mu1 - mu2
    evals, evecs = torch.linalg.eigh(sigma1)
    sqrt_s1 = evecs @ torch.diag(evals.clamp_min(0).sqrt()) @ evecs.T
    m = sqrt_s1 @ sigma2 @ sqrt_s1
    tr_covmean = torch.linalg.eigvalsh((m + m.T) / 2).clamp_min(0).sqrt().sum()
    fid = diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def fid_from_features(real_feats: torch.Tensor, fake_feats: torch.Tensor) -> float:
    return frechet_distance(*feature_stats(real_feats), *feature_stats(fake_feats))


@torch.no_grad()
def mode_coverage(fidnet, images: torch.Tensor, device, n_classes: int = 10, batch: int = 500) -> dict:
    """Ukur keberagaman mode: distribusi kelas prediksi classifier pada sampel sintetis.

    Entropi ternormalisasi = H(hist) / log(K). Nilai 1.0 = semua kelas muncul merata,
    nilai rendah = mode collapse (sampel menumpuk di sedikit kelas).
    """
    fidnet.eval()
    probs = torch.cat([F.softmax(fidnet(images[i:i + batch].to(device)), 1).cpu()
                       for i in range(0, len(images), batch)])
    hist = torch.bincount(probs.argmax(1), minlength=n_classes).float()
    p = hist / hist.sum()
    entropy = -(p[p > 0] * p[p > 0].log()).sum().item()
    return dict(class_hist=hist.int().tolist(),
                norm_entropy=max(0.0, entropy / math.log(n_classes)),
                classes_covered=int((p >= 0.01).sum()),
                mean_confidence=probs.max(1).values.mean().item())


# ------------------------------------------------------------------- latensi
@torch.no_grad()
def measure_latency(fn, device, n_warmup: int = 2, n_runs: int = 10) -> dict:
    """Latensi (ms) satu panggilan fn(); warm-up dulu agar inisialisasi CUDA tidak ikut terukur."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return dict(ms=t.mean().item(), ms_std=t.std().item() if n_runs > 1 else 0.0, runs=n_runs)
