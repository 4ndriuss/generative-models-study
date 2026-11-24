# Studi Komparatif Model Generatif (AE, Sparse AE, VAE, GAN, DCGAN, DDPM)

Implementasi latihan untuk tugas _Deep Learning D6421 – Sesi 07_. Isinya:
enam model generatif, eksperimen batasan MLP Classifier (Bagian 2), evaluasi Test Set
(MSE, PSNR, SSIM, FID, latensi, mode coverage), dan lima figur wajib.

## Struktur

```
src/
  config.py          semua hyperparameter (satu sumber kebenaran untuk tabel konfigurasi)
  data.py            Fashion-MNIST/MNIST/KMNIST, split 80/10/10 dengan seed 42
  metrics.py         MSE, PSNR, SSIM, FID (Frechet), mode coverage, latensi
  training.py        loop training umum + loop GAN, checkpoint, riwayat loss
  plotting.py        helper matplotlib
  models/
    autoencoders.py  VanillaAE, SparseAE, VAE (encoder 784-256-64, decoder 64-256-784)
    gans.py          Vanilla GAN (MLP) dan DCGAN
    ddpm.py          DDPM + U-Net kecil + EMA
    classifiers.py   MLP Classifier 784-256-64-10 dan FIDNet
train.py             latih satu / semua model
mlp_experiment.py    Bagian 2 (Tahap 1-5) + Figur 3
evaluate.py          tabel metrik Bagian IV
make_figures.py      Figur 1, 2, 4, 5 (+ figur tambahan)
run_all.py           seluruh pipeline berurutan
```

## Menjalankan

```bash
pip install -r requirements.txt

# uji cepat pipeline (data sintetis, tanpa unduhan, ~2 menit di CPU) -> outputs_smoke/
python run_all.py --fake-data --quick

# eksperimen penuh -> outputs/
python run_all.py
```

Atau per langkah:

```bash
python train.py --model fidnet      # wajib duluan: dipakai untuk FID & mode coverage
python train.py --model mlp
python train.py --model ae
python train.py --model sae --sparsity-weight 5e-3
python train.py --model vae
python train.py --model gan         # --g-loss saturating untuk melihat vanishing gradient
python train.py --model dcgan
python train.py --model ddpm
python mlp_experiment.py
python evaluate.py
python make_figures.py
```

**Hardware.** DDPM butuh GPU. Di CPU satu epoch DDPM sekitar 50+ menit, dan sampling 1000
langkah untuk FID bisa makan berjam-jam. Di Google Colab (GPU T4):
`Runtime → Change runtime type → T4 GPU`, upload folder ini, lalu `!python run_all.py`.
Perkiraan total di T4 sekitar 1–1,5 jam, terutama untuk DDPM dan sampling FID-nya.

## Di mana hasilnya untuk laporan

| Bagian laporan             | File                                                                            |
| -------------------------- | ------------------------------------------------------------------------------- |
| II – tabel konfigurasi     | `src/config.py`, `outputs/<model>/model.pt` (`model_kwargs`, `train_cfg`)       |
| III – batasan MLP          | `outputs/mlp_experiment/summary.md`, `results.json`, `fig3_mlp_limitations.png` |
| IV – tabel metrik          | `outputs/results/metrics_table.md`, `metrics.json`                              |
| V – figur 1, 2, 4, 5       | `outputs/figures/`                                                              |
| Q3 – progres GAN per epoch | `outputs/gan/progress/`, `outputs/dcgan/progress/`                              |

## Keputusan desain (perlu disebut di laporan)

- **Kerangka sama.** AE, SAE, VAE, dan MLP Classifier memakai 784→256→64, dan semua decoder
  64→256→784. Jadi perbedaan hasil berasal dari _fungsi objektif_, bukan kapasitas.
- **Loss rekonstruksi.** AE/SAE memakai MSE (jumlah per piksel, rata-rata per batch), sama dengan
  decoder Tahap 4. VAE memakai BCE + KL (likelihood Bernoulli, formulasi Kingma & Welling).
- **Sparse AE.** Penalti λ‖h‖₁ dikenakan pada hidden layer 256-d; bottleneck tetap 64.
  λ = 5e-3 belum di-tuning, jadi coba sweep {1e-3, 5e-3, 1e-2} untuk melihat trade-off.
- **FID.** Memakai fitur 128-d dari _FIDNet_ (CNN yang dilatih pada data train), bukan InceptionV3,
  karena citranya grayscale 28×28. Angkanya hanya sebanding antarmodel di eksperimen ini.
  Semua model memakai jumlah sampel yang sama (5000) vs seluruh Test Set.
- **FID\* untuk AE/SAE.** Template menulis N/A karena AE tidak punya prior. Sebagai bukti tambahan
  _latent voids_, laten train di-fit dengan N(μ, Σ) lalu di-sampling.
- **Latensi** diukur setelah warm-up dengan `cuda.synchronize()`, batch 64. DDPM = T = 1000 langkah.
- **Reproduktibilitas.** `set_seed(42)` di setiap skrip, split dengan generator ber-seed, dan
  `cudnn.deterministic = True`.
