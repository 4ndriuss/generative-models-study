"""MLP Classifier (Bagian 2) dan FIDNet (feature extractor untuk FID)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """784 -> 256 -> 64 -> 10, dilatih dengan Cross-Entropy (memodelkan P(y | x)).

    Strukturnya identik dengan encoder Vanilla AE, tapi tujuannya berbeda: CE hanya
    menghargai informasi yang membedakan kelas, sehingga variasi intra-kelas boleh dibuang.
    Bobot nn.Linear berbentuk (out, in): fc1.weight = W1 (256x784), fc2.weight = W2 (64x256),
    fc3.weight = W3 (10x64).
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def features(self, x):
        """z_MLP in R^64: representasi tepat sebelum lapisan klasifikasi."""
        h1 = F.relu(self.fc1(x.flatten(1)))
        return F.relu(self.fc2(h1))

    def forward(self, x):
        return self.fc3(self.features(x))

    def loss(self, x, y):
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        return loss, {"loss": loss.item(), "acc": acc.item()}


class FIDNet(nn.Module):
    """CNN kecil yang dilatih pada data train untuk menghasilkan fitur 128-d.

    Kenapa bukan InceptionV3? Inception dilatih pada ImageNet RGB 299x299; untuk citra
    grayscale 28x28 praktik yang umum adalah memakai classifier domain-spesifik. Konsekuensinya,
    angka FID di sini hanya bisa dibandingkan ANTARMODEL dalam eksperimen ini, bukan dengan
    angka FID di paper.
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, feat_dim), nn.ReLU())
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(feat_dim, 10)

    def features(self, x):
        return self.fc(self.conv(x))

    def forward(self, x):
        return self.head(self.drop(self.features(x)))

    def loss(self, x, y):
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        return loss, {"loss": loss.item(), "acc": acc.item()}
