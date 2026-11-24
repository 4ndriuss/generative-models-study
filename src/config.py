SEED = 42
DATASET = "fashionmnist"      # pilihan: fashionmnist | mnist | kmnist
BATCH_SIZE = 128
FID_SAMPLES = 5000            # jumlah sampel sintetis per model untuk FID (sama untuk semua model)
LATENCY_BATCH = 64            # tugas meminta latensi per batch 64 sampel

# Dua kelas kontras untuk Figur 4 (interpolasi). Fashion-MNIST: 1 = Trouser, 9 = Ankle boot.
INTERP_CLASS_A = 1
INTERP_CLASS_B = 9
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# kwargs   -> argumen konstruktor model
# epochs   -> default untuk GPU (Colab T4). Di CPU, kurangi epoch GAN/DCGAN/DDPM.
# lr/betas -> optimizer Adam
MODEL_CONFIGS = {
    # CNN kecil, dipakai sebagai feature extractor FID dan pengukur mode coverage
    "fidnet": dict(kwargs={}, epochs=10, lr=1e-3),
    # MLP classifier 784 -> 256 -> 64 -> 10 (Bagian 2)
    "mlp": dict(kwargs={}, epochs=20, lr=1e-3),
    "ae": dict(kwargs=dict(latent_dim=64, hidden_dim=256), epochs=30, lr=1e-3),
    "sae": dict(kwargs=dict(latent_dim=64, hidden_dim=256, sparsity_weight=5e-3), epochs=30, lr=1e-3),
    "vae": dict(kwargs=dict(latent_dim=64, hidden_dim=256, beta=1.0), epochs=30, lr=1e-3),
    "gan": dict(kwargs=dict(arch="mlp", z_dim=100, g_loss="nonsat"), epochs=100, lr=2e-4, betas=(0.5, 0.999)),
    "dcgan": dict(kwargs=dict(arch="dcgan", z_dim=100, g_loss="nonsat"), epochs=30, lr=2e-4, betas=(0.5, 0.999)),
    "ddpm": dict(kwargs=dict(timesteps=1000, base_ch=64, ema_decay=0.999), epochs=40, lr=2e-4, grad_clip=1.0),
}

GENERATIVE_MODELS = ["ae", "sae", "vae", "gan", "dcgan", "ddpm"]
TRAIN_ORDER = ["fidnet", "mlp", "ae", "sae", "vae", "gan", "dcgan", "ddpm"]

# Mode --quick: untuk uji cepat pipeline (bukan untuk hasil laporan)
QUICK = dict(train_subset=512, epochs=1, ddpm_timesteps=20, fid_samples=200)
