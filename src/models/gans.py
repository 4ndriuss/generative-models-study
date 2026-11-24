"""Vanilla GAN (fully connected) dan DCGAN (fully convolutional).

Objektif minimax (Goodfellow et al., 2014):
    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
Untuk D optimal, objektif G setara dengan meminimalkan 2 * JS(P_data || P_g) - log 4.

Kedua varian di sini memakai loss, optimizer, dan prior z ~ N(0, I) yang sama;
yang berbeda hanya arsitekturnya. Jadi perbedaan stabilitas = efek inductive bias arsitektur.
"""
import torch
import torch.nn as nn

from ..utils import to_unit

IMG_DIM = 28 * 28


# ------------------------------------------------------------- Vanilla (MLP)
class MLPGenerator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, IMG_DIM), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    """Tanpa BatchNorm: setiap piksel diperlakukan sebagai fitur independen (tanpa prior spasial)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_DIM, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),                      # logit; sigmoid ada di dalam BCEWithLogits
        )

    def forward(self, x):
        return self.net(x).view(-1)


# -------------------------------------------------------------------- DCGAN
class DCGenerator(nn.Module):
    """Prinsip DCGAN (Radford et al., 2016): tanpa pooling/FC tersembunyi, upsampling via
    transposed conv, BatchNorm di G, ReLU di G dan Tanh di output.
    z (100x1x1) -> 256x7x7 -> 128x14x14 -> 1x28x28
    """

    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 7, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z.view(-1, self.z_dim, 1, 1))


class DCDiscriminator(nn.Module):
    """Strided conv sebagai downsampling, LeakyReLU, BatchNorm kecuali di layer input.
    1x28x28 -> 64x14x14 -> 128x7x7 -> logit
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 7, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def dcgan_weights_init(m: nn.Module) -> None:
    """Inisialisasi bobot N(0, 0.02) sesuai paper DCGAN."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ----------------------------------------------------------------- wrapper
class GAN(nn.Module):
    """Pembungkus G + D. arch: 'mlp' (Vanilla GAN) atau 'dcgan'.

    g_loss:
      'nonsat'     -> G meminimalkan -log D(G(z))  (non-saturating, dipakai dalam praktik)
      'saturating' -> G meminimalkan log(1 - D(G(z))) (minimax asli; gradien hilang saat D menang)
    """

    def __init__(self, arch: str = "mlp", z_dim: int = 100, g_loss: str = "nonsat"):
        super().__init__()
        self.arch, self.z_dim, self.g_loss = arch, z_dim, g_loss
        self.latent_dim = z_dim
        if arch == "mlp":
            self.G, self.D = MLPGenerator(z_dim), MLPDiscriminator()
        elif arch == "dcgan":
            self.G, self.D = DCGenerator(z_dim), DCDiscriminator()
            self.apply(dcgan_weights_init)
        else:
            raise ValueError(arch)

    def decode(self, z):
        """z -> gambar [0, 1]."""
        return to_unit(self.G(z))

    @torch.no_grad()
    def sample(self, n: int, device):
        return self.decode(torch.randn(n, self.z_dim, device=device))
