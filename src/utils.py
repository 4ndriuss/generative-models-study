"""Utilitas umum: seed, device, konversi rentang piksel, I/O, dan timer."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def set_seed(seed: int = 42) -> None:
    """Kunci semua sumber keacakan agar eksperimen bisa direproduksi."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_signed(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] -> [-1, 1]. GAN/DCGAN/DDPM bekerja di rentang ini (output tanh)."""
    return x * 2 - 1


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]. Semua metrik dan figur dihitung di rentang [0, 1]."""
    return ((x + 1) / 2).clamp(0, 1)


def set_output_root(path: str | Path) -> None:
    global _OUTPUT_ROOT
    p = Path(path)
    _OUTPUT_ROOT = p if p.is_absolute() else PROJECT_ROOT / p


def output_dir(*parts: str) -> Path:
    p = _OUTPUT_ROOT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sync(device: torch.device) -> None:
    """Tunggu kernel GPU selesai; wajib sebelum mengukur waktu di CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize()


class Timer:
    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed = 0.0

    def __enter__(self):
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.elapsed = time.perf_counter() - self._t0
