"""Helper matplotlib untuk figur laporan."""
import matplotlib

matplotlib.use("Agg")      # render tanpa layar (aman di server/Colab)
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import make_grid  # noqa: E402

DPI = 200


def grid_image(images: torch.Tensor, nrow: int, padding: int = 2):
    """(N,1,H,W) di [0,1] -> array HxW untuk imshow."""
    g = make_grid(images.detach().cpu().clamp(0, 1), nrow=nrow, padding=padding, pad_value=1.0)
    return g[0].numpy()


def show_grid(ax, images, nrow, title=None, fontsize=10):
    ax.imshow(grid_image(images, nrow), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fontsize)


def row_figure(rows, title, path, col_labels=None, row_height=1.25):
    """rows: list of (label, images (N,1,H,W)). Setiap baris = satu strip gambar berlabel."""
    n_cols = max(len(imgs) for _, imgs in rows)
    fig, axes = plt.subplots(len(rows), 1, figsize=(1.1 * n_cols + 2.2, row_height * len(rows) + 0.8))
    axes = [axes] if len(rows) == 1 else axes
    for ax, (label, imgs) in zip(axes, rows):
        show_grid(ax, imgs, nrow=len(imgs))
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
    if col_labels:
        w = 28 + 2
        axes[0].set_xticks([2 + w * i + 14 for i in range(len(col_labels))])
        axes[0].set_xticklabels(col_labels, fontsize=9)
        axes[0].xaxis.tick_top()
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
