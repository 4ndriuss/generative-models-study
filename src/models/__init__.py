"""Registry model: nama singkat -> kelas."""
from .autoencoders import VAE, MLPDecoder, SparseAE, VanillaAE
from .classifiers import FIDNet, MLPClassifier
from .ddpm import DDPM
from .gans import GAN

MODEL_CLASSES = {
    "fidnet": FIDNet,
    "mlp": MLPClassifier,
    "ae": VanillaAE,
    "sae": SparseAE,
    "vae": VAE,
    "gan": GAN,
    "dcgan": GAN,
    "ddpm": DDPM,
}

DISPLAY_NAMES = {
    "ae": "Vanilla Autoencoder",
    "sae": "Sparse Autoencoder",
    "vae": "Variational Autoencoder (VAE)",
    "gan": "Vanilla GAN (MLP)",
    "dcgan": "DCGAN",
    "ddpm": "DDPM",
    "mlp": "MLP Classifier",
    "fidnet": "FIDNet",
}


def build_model(name: str, **kwargs):
    return MODEL_CLASSES[name](**kwargs)


__all__ = ["build_model", "MODEL_CLASSES", "DISPLAY_NAMES", "MLPDecoder", "VanillaAE",
           "SparseAE", "VAE", "GAN", "DDPM", "MLPClassifier", "FIDNet"]
