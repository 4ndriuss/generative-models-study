"""Argumen CLI yang dipakai bersama oleh semua skrip."""
import argparse

from . import config
from .utils import get_device, set_output_root


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", default=config.DATASET, choices=["fashionmnist", "mnist", "kmnist"])
    p.add_argument("--device", default=None, help="cuda / cpu (default: otomatis)")
    p.add_argument("--fake-data", action="store_true",
                   help="pakai data sintetis tanpa unduhan (hanya untuk uji pipeline)")
    p.add_argument("--quick", action="store_true",
                   help="uji cepat: subset kecil, 1 epoch, DDPM T=20, sampel FID sedikit")
    p.add_argument("--output-dir", default=None, help="folder output (default: outputs/)")
    return p


def setup(args):
    """Terapkan folder output dan kembalikan device."""
    if args.output_dir:
        set_output_root(args.output_dir)
    elif args.quick or args.fake_data:
        set_output_root("outputs_smoke")      # jangan menimpa hasil training sungguhan
    return get_device(args.device)


def data_kwargs(args) -> dict:
    return dict(dataset=args.dataset, fake=args.fake_data, quick=args.quick)
