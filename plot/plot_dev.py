#plot/plot_dev.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    acc = np.load(Path(args.hist))
    wps = np.load(Path(args.wps))

    # -- safeguard ---------------------------------------------------------
    if len(acc) != len(wps):
        print(f"[plot_dev] ⚠  length mismatch acc={len(acc)} vs wps={len(wps)} – trimming.")
        k = min(len(acc), len(wps))
        acc, wps = acc[:k], wps[:k]

    epochs = np.arange(len(acc))

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(epochs, acc, marker='o', lw=2, label='Dev char-acc')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dev char-accuracy')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, wps, '--', color='grey', label='Words/sec')
    ax2.set_ylabel('Words per second', color='grey')
    ax2.tick_params(axis='y', colors='grey')

    # combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right')

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Plot saved to {Path(args.out).resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", default="experiments/dev_acc_history.npy")
    ap.add_argument("--wps",  default="experiments/wps_history.npy")
    ap.add_argument("--out",  default="docs/dev_accuracy_wps.png")
    main(ap.parse_args())