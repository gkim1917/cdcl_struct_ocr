#plot/plot_dev.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(args):
    hist_path = Path(args.hist)
    if not hist_path.is_file():
        raise FileNotFoundError(f"Cannot find history file: {hist_path}")

    acc = np.load(hist_path)  # 1-D array of dev accuracies
    epochs = np.arange(len(acc))

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, acc, marker="o", lw=1.8)
    plt.title("Greedy dev accuracy vs. epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Dev char-accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Plot saved to {out_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hist",
        default="experiments/dev_acc_history.npy",
        help="Path to the saved NumPy array with dev accuracies",
    )
    ap.add_argument(
        "--out",
        default="docs/dev_accuracy.png",
        help="Where to write the PNG plot",
    )
    main(ap.parse_args())
