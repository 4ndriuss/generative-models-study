"""Dataset, split 80/10/10, dan DataLoader terstandarisasi.

Gambar disimpan di rentang [0, 1] (bentuk N x 1 x 28 x 28). Model yang butuh [-1, 1]
(GAN, DCGAN, DDPM) mengonversi sendiri di dalam fungsi loss-nya, sehingga semua model
melihat data yang persis sama.
"""
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets

from . import config
from .utils import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"

DATASETS = {
    "fashionmnist": datasets.FashionMNIST,
    "mnist": datasets.MNIST,
    "kmnist": datasets.KMNIST,
}

CLASS_NAMES = {
    "fashionmnist": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "mnist": [str(i) for i in range(10)],
    "kmnist": ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"],
}


def _load_full(name: str) -> TensorDataset:
    """Gabungkan split train (60k) + test (10k) resmi = 70k citra, lalu kita split ulang 80/10/10."""
    cls = DATASETS[name]
    parts = [cls(DATA_DIR, train=t, download=True) for t in (True, False)]
    x = torch.cat([p.data for p in parts]).float().div(255).unsqueeze(1)
    y = torch.cat([torch.as_tensor(p.targets) for p in parts]).long()
    return TensorDataset(x, y)


def _fake_full(n: int = 3000) -> TensorDataset:
    """Data sintetis berpola per kelas. HANYA untuk uji cepat pipeline tanpa unduhan."""
    g = torch.Generator().manual_seed(0)
    templates = (torch.rand(10, 1, 28, 28, generator=g) > 0.5).float()
    y = torch.randint(0, 10, (n,), generator=g)
    x = (0.7 * templates[y] + 0.3 * torch.rand(n, 1, 28, 28, generator=g)).clamp(0, 1)
    return TensorDataset(x, y)


def get_splits(dataset: str = config.DATASET, fake: bool = False, seed: int = config.SEED):
    full = _fake_full() if fake else _load_full(dataset)
    n = len(full)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(seed)   # split identik untuk semua model
    return random_split(full, [n_train, n_val, n_test], generator=gen)


def subset_tensors(subset: Subset):
    """Ambil tensor (x, y) dari sebuah Subset secara langsung (cepat untuk evaluasi)."""
    x, y = subset.dataset.tensors
    idx = torch.as_tensor(subset.indices)
    return x[idx], y[idx]


def get_loaders(batch_size: int = config.BATCH_SIZE, dataset: str = config.DATASET,
                fake: bool = False, train_subset: int | None = None):
    train, val, test = get_splits(dataset, fake)
    if train_subset:
        train = Subset(train, list(range(min(train_subset, len(train)))))
    g = torch.Generator().manual_seed(config.SEED)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
