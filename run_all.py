"""Jalankan seluruh pipeline secara berurutan.

    python run_all.py                       # eksperimen penuh (sebaiknya di GPU)
    python run_all.py --fake-data --quick   # uji pipeline cepat tanpa unduhan dataset
Argumen tambahan (--dataset, --device, --output-dir) diteruskan ke tiap skrip.
"""
import subprocess
import sys

STEPS = [
    ["train.py", "--model", "all"],
    ["mlp_experiment.py"],
    ["evaluate.py"],
    ["make_figures.py"],
]


def main():
    extra = sys.argv[1:]
    for step in STEPS:
        cmd = [sys.executable, *step, *extra]
        print("\n>>> " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
