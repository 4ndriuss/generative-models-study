"""Denoising Diffusion Probabilistic Model (Ho et al., 2020) dengan U-Net kecil untuk 28x28.

Forward process (tetap, tanpa parameter):
    q(x_t | x_0) = N( sqrt(abar_t) x_0, (1 - abar_t) I ),  abar_t = prod_{s<=t} (1 - beta_s)
Model eps_theta(x_t, t) dilatih memprediksi derau yang ditambahkan:
    L_simple = E_{t, x_0, eps} || eps - eps_theta( sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, t ) ||^2
Ini regresi terawasi biasa (target eps diketahui) -> jauh lebih stabil daripada minimax GAN.
Harganya: sampling butuh T langkah jaringan berurutan.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import to_signed, to_unit


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Embedding sinusoidal (seperti positional encoding Transformer) untuk indeks waktu t."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(temb))[:, :, None, None]   # informasi "seberapa berisik" x_t
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).transpose(2, 3).unbind(1)
        out = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    """U-Net 3 resolusi: 28x28 (ch) -> 14x14 (2ch) -> 7x7 (2ch, + self-attention)."""

    def __init__(self, base_ch: int = 64):
        super().__init__()
        c1, c2 = base_ch, base_ch * 2
        t_dim = base_ch * 4
        self.base_ch = base_ch
        self.time_mlp = nn.Sequential(nn.Linear(base_ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inc = nn.Conv2d(1, c1, 3, padding=1)
        self.down0 = ResBlock(c1, c1, t_dim)
        self.ds0 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)            # 28 -> 14
        self.down1 = ResBlock(c1, c2, t_dim)
        self.ds1 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)            # 14 -> 7
        self.down2 = ResBlock(c2, c2, t_dim)
        self.mid1 = ResBlock(c2, c2, t_dim)
        self.mid_attn = SelfAttention(c2)
        self.mid2 = ResBlock(c2, c2, t_dim)
        self.up2 = ResBlock(c2 + c2, c2, t_dim)
        self.us2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c2, 3, padding=1))
        self.up1 = ResBlock(c2 + c2, c2, t_dim)
        self.us1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c2, 3, padding=1))
        self.up0 = ResBlock(c2 + c1, c1, t_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.base_ch))
        h0 = self.inc(x)
        s1 = self.down0(h0, temb)                      # 28x28 skip
        s2 = self.down1(self.ds0(s1), temb)            # 14x14 skip
        s3 = self.down2(self.ds1(s2), temb)            # 7x7 skip
        m = self.mid2(self.mid_attn(self.mid1(s3, temb)), temb)
        u = self.up2(torch.cat([m, s3], 1), temb)
        u = self.up1(torch.cat([self.us2(u), s2], 1), temb)
        u = self.up0(torch.cat([self.us1(u), s1], 1), temb)
        return self.out(u)


class DDPM(nn.Module):
    def __init__(self, timesteps: int = 1000, base_ch: int = 64, beta_start: float = 1e-4,
                 beta_end: float = 0.02, ema_decay: float = 0.999):
        super().__init__()
        self.T = timesteps
        self.ema_decay = ema_decay
        self.net = UNet(base_ch)
        # EMA bobot: rata-rata bergerak parameter, menghasilkan sampel lebih halus/stabil
        self.ema = copy.deepcopy(self.net).requires_grad_(False)

        # Jadwal linear disesuaikan bila T kecil (mode --quick) agar abar_T tetap mendekati 0
        scale = 1000 / timesteps
        betas = torch.linspace(beta_start * scale, min(beta_end * scale, 0.999), timesteps)
        alphas = 1 - betas
        abar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        self.register_buffer("sqrt_abar", abar.sqrt())
        self.register_buffer("sqrt_1m_abar", (1 - abar).sqrt())

    # ------------------------------------------------------------ training
    def q_sample(self, x0, t, noise):
        """Loncat langsung ke x_t dari x_0 (bentuk tertutup forward process)."""
        return self.sqrt_abar[t].view(-1, 1, 1, 1) * x0 + self.sqrt_1m_abar[t].view(-1, 1, 1, 1) * noise

    def loss(self, x, y=None):
        x0 = to_signed(x)
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        pred = self.net(self.q_sample(x0, t, noise), t)
        loss = F.mse_loss(pred, noise)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def update_ema(self):
        for pe, p in zip(self.ema.parameters(), self.net.parameters()):
            pe.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)
        for be, b in zip(self.ema.buffers(), self.net.buffers()):
            be.copy_(b)

    # ------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, n: int, device, use_ema: bool = True, trajectory_steps=None):
        """Reverse process ancestral sampling, x_T ~ N(0, I) -> x_0:
            x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - abar_t) * eps_theta(x_t, t)) + sigma_t z
        dengan sigma_t^2 = beta_t. Butuh T evaluasi jaringan berurutan.

        trajectory_steps: daftar t yang ingin direkam (untuk Figur 5). Bila diberikan,
        fungsi mengembalikan (sampel, {t: gambar}).
        """
        net = self.ema if use_ema else self.net
        net.eval()
        x = torch.randn(n, 1, 28, 28, device=device)
        traj = {}
        if trajectory_steps is not None and self.T in trajectory_steps:
            traj[self.T] = to_unit(x.clamp(-1, 1)).cpu()
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps = net(x, tt)
            coef = self.betas[t] / self.sqrt_1m_abar[t]
            mean = (x - coef * eps) / self.alphas[t].sqrt()
            x = mean + self.betas[t].sqrt() * torch.randn_like(x) if t > 0 else mean
            if trajectory_steps is not None and t in trajectory_steps:
                traj[t] = to_unit(x.clamp(-1, 1)).cpu()
        out = to_unit(x.clamp(-1, 1))
        return (out, traj) if trajectory_steps is not None else out
