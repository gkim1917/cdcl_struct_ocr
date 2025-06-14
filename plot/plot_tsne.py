#!/usr/bin/env python
"""
plot_tsne.py  –  t-SNE visualisation of OCR pixel features.
Produces 'char_tsne.pdf' in the location you specify with --out.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def load_raw(csv_path: Path):
    """Load the gzipped UW-III CSV exactly like dataset.py."""
    # column names copied from dataset.py so we can extract the pixel_* cols
    cols = (['letter_id', 'letter', 'next_id', 'word_id', 'position', 'fold'] +
            [f'stat_{i}' for i in range(16)] +
            [f'pixel_{i}' for i in range(112)])
    df = pd.read_csv(csv_path, compression="gzip", sep=r"\s+", header=None,
                     comment="#")
    df.columns = cols
    return df


def main(args):
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = load_raw(csv_path)
    pixel_cols = [c for c in df.columns if c.startswith("pixel_")]

    # ---------- subsample for speed ----------------------------------
    rng = np.random.default_rng(42)
    keep_idx = rng.choice(len(df), size=args.n_samples, replace=False)
    X = df.loc[keep_idx, pixel_cols].to_numpy(np.float32)
    labels = df.loc[keep_idx, "letter"].str.lower().to_numpy()

    # ---------- t-SNE ------------------------------------------------
    emb = TSNE(init="pca",
               n_components=2,
               perplexity=30,
               learning_rate="auto",
               random_state=0).fit_transform(X)

    # ---------- plot -------------------------------------------------
    plt.figure(figsize=(4.8, 4.2))
    plt.scatter(emb[:, 0], emb[:, 1], s=6, alpha=.55)

    for i in rng.choice(len(emb), size=40, replace=False):
        plt.text(emb[i, 0], emb[i, 1], labels[i],
                 fontsize=6, alpha=.8)

    plt.xticks([]); plt.yticks([])
    plt.title("t-SNE of normalised pixel features")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved figure → {out_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default="data/letter.data.gz",
                    help="Path to the gzipped UW-III CSV")
    ap.add_argument("--out",
                    default="docs/char_tsne.pdf",
                    help="Output file (PDF/PNG)")
    ap.add_argument("--n-samples", type=int, default=3000,
                    help="characters to subsample for t-SNE")
    main(ap.parse_args())
