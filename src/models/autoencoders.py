"""Vanilla Autoencoder, Sparse Autoencoder, dan Variational Autoencoder (MLP).

Ketiganya memakai kerangka yang SAMA dengan MLP classifier di Bagian 2:
    encoder 784 -> 256 -> 64,   decoder 64 -> 256 -> 784 (sigmoid)
sehingga perbedaan hasil murni berasal dari fungsi objektif, bukan dari kapasitas jaringan.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_DIM = 28 * 28


class MLPDecoder(nn.Module):
    """64 -> 256 -> 784. Dipakai ulang oleh AE, SAE, VAE, dan frozen-latent decoder (Tahap 4)."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, IMG_DIM), nn.Sigmoid(),   # piksel di [0, 1]
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class VanillaAE(nn.Module):
    """Loss: L = ||x - x_hat||^2 (dijumlah per piksel, dirata-rata per batch).

    Encoder deterministik: tiap x dipetakan ke SATU titik z. Tidak ada yang memaksa
    titik-titik ini mengisi ruang laten secara rapat -> muncul "latent voids".
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_hidden = nn.Sequential(nn.Flatten(), nn.Linear(IMG_DIM, hidden_dim), nn.ReLU())
        self.enc_code = nn.Linear(hidden_dim, latent_dim)     # bottleneck linear
        self.decoder = MLPDecoder(latent_dim, hidden_dim)

    def hidden(self, x):
        """Aktivasi hidden layer 256-d (dipakai untuk analisis sparsity)."""
        return self.enc_hidden(x)

    def encode(self, x):
        return self.enc_code(self.enc_hidden(x))

    def decode(self, z):
        return self.decoder(z)

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    def forward(self, x):
        return self.reconstruct(x)

    def loss(self, x, y=None):
        recon = self(x)
        rec = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        return rec, {"loss": rec.item(), "recon": rec.item()}


class SparseAE(VanillaAE):
    """Loss: L = ||x - x_hat||^2 + lambda * ||h||_1, dengan h = aktivasi hidden 256-d.

    Penalti L1 mendorong sebagian besar unit h bernilai nol untuk tiap input, sehingga
    tiap unit hanya aktif untuk pola lokal tertentu (detektor fitur yang lebih "terspesialisasi").
    Harganya: kapasitas efektif turun -> MSE rekonstruksi biasanya sedikit naik.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, sparsity_weight: float = 5e-3):
        super().__init__(latent_dim, hidden_dim)
        self.sparsity_weight = sparsity_weight

    def loss(self, x, y=None):
        h = self.hidden(x)
        recon = self.decode(self.enc_code(h))
        rec = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        l1 = h.abs().sum(1).mean()
        total = rec + self.sparsity_weight * l1
        return total, {"loss": total.item(), "recon": rec.item(), "l1": l1.item()}


class VAE(nn.Module):
    """Loss (negatif ELBO):
        L = E_q[-log p(x|z)] + beta * KL( q(z|x) || N(0, I) )
    dengan q(z|x) = N(mu(x), diag(sigma^2(x))) dan p(x|z) Bernoulli (-> BCE).

    KL memaksa posterior tiap data tumpang-tindih dengan prior N(0, I), sehingga ruang
    laten menjadi rapat dan kontinu -> sampling dari prior dan interpolasi menjadi bermakna.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.enc_hidden = nn.Sequential(nn.Flatten(), nn.Linear(IMG_DIM, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLPDecoder(latent_dim, hidden_dim)

    def encode_dist(self, x):
        h = self.enc_hidden(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode(self, x):
        """Representasi deterministik = mean posterior (untuk rekonstruksi & interpolasi)."""
        return self.encode_dist(x)[0]

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps, eps ~ N(0, I): sampling dibuat dapat diturunkan terhadap mu, sigma."""
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(z)

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def loss(self, x, y=None):
        recon, mu, logvar = self(x)
        b = x.size(0)
        rec = F.binary_cross_entropy(recon, x, reduction="sum") / b
        # KL analitik antara N(mu, sigma^2) dan N(0, 1), dijumlah per dimensi laten
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
        total = rec + self.beta * kl
        return total, {"loss": total.item(), "recon": rec.item(), "kl": kl.item()}

    @torch.no_grad()
    def sample(self, n: int, device):
        return self.decode(torch.randn(n, self.latent_dim, device=device))
