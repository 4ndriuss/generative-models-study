"""Latih satu atau semua model.

Contoh:
    python train.py --model vae
    python train.py --model ddpm --epochs 20
    python train.py --model all
    python train.py --model all --fake-data --quick     # uji pipeline tanpa unduhan
"""
from src import config
from src.cli import common_parser, data_kwargs, setup
from src.training import train_model


def main():
    p = common_parser("Training model generatif")
    p.add_argument("--model", required=True, choices=config.TRAIN_ORDER + ["all"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, choices=[128, 256])
    p.add_argument("--sparsity-weight", type=float, default=None, help="lambda L1 Sparse AE")
    p.add_argument("--g-loss", default=None, choices=["nonsat", "saturating"], help="objektif G pada GAN")
    p.add_argument("--timesteps", type=int, default=None, help="T pada DDPM")
    args = p.parse_args()
    device = setup(args)

    names = config.TRAIN_ORDER if args.model == "all" else [args.model]
    for name in names:
        train_model(name, device, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
                    sparsity_weight=args.sparsity_weight, g_loss=args.g_loss,
                    timesteps=args.timesteps, **data_kwargs(args))


if __name__ == "__main__":
    main()
