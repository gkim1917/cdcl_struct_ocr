# train_crf.py
#
# A compact CRF baseline
# Uses the identical OCRDataset loader, converts each (720-D) window
# vector into a sparse feature-dict { "f0":v0 , "f1":v1 , ... } so that
# sklearn-crfsuite can digest it.
#
#  ▸ Training time  ≈ 60 s on 2 500 words (↔ 18 k characters)
#  ▸ Dev accuracy   printed at the end
#  ▸ Trained model  pickled to experiments/crf_model.pkl

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn_crfsuite import CRF, metrics

from src.dataset import OCRDataset   # adjust import path if needed

# ----------------------- helpers -------------------------------------
def vec2dict(vec):
    """720-D -> {'f0':v0, …}.  Using str keys keeps CRF memory light."""
    return {f"f{i}": float(v) for i, v in enumerate(vec)}

def seq2features(X):
    """X: (n,d) ndarray  → list of n dicts"""
    return [vec2dict(row) for row in X]

def seq2labels(y):
    """y: 1-D int array 0–25 → list of single-char strings"""
    return [chr(ord("a") + int(t)) for t in y]

# ----------------------- main ----------------------------------------
def main(args):
    # ----- load splits -------------------------------------------------
    train_ds = OCRDataset(split="train",
                          n_train=args.n_train,
                          window=args.window)
    dev_ds   = OCRDataset(split="test",
                          n_train=args.n_train,
                          window=args.window,
                          mu_sigma=(train_ds.mu, train_ds.sig))

    # ----- build lists for sklearn-crfsuite ----------------------------
    X_train = [seq2features(w["X"])    for w in train_ds.words]
    y_train = [seq2labels (w["y"])     for w in train_ds.words]
    X_dev   = [seq2features(w["X"])    for w in dev_ds.words]
    y_dev   = [seq2labels (w["y"])     for w in dev_ds.words]

    # ----- set up CRF --------------------------------------------------
    crf = CRF(
        algorithm="lbfgs",
        max_iterations=args.max_iter,
        all_possible_transitions=True,
        # L2 regulariser roughly matched to perceptron LR=0.15
        c2=1e-2,
    )

    print(f"Training CRF on {len(X_train)} words …")
    crf.fit(X_train, y_train)
    print("Done.")

    # ----- evaluate ----------------------------------------------------
    y_pred = crf.predict(X_dev)
    # char accuracy
    correct = total = 0
    for yp, yt in zip(y_pred, y_dev):
        correct += sum(p == t for p, t in zip(yp, yt))
        total   += len(yt)
    acc = correct / total
    print(f"\nCRF dev char-accuracy: {acc:.4f}")

    # ----- save model --------------------------------------------------
    out = Path("experiments") / "crf_model.pkl"
    out.parent.mkdir(exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(crf, f)
    print(f"Pickled model → {out.resolve()}")

# ----------------------- CLI -----------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=2500,
                   help="words in training subset (≃ HW spec)")
    p.add_argument("--window",  type=int, default=5,
                   help="symmetric context window (same as main model)")
    p.add_argument("--max_iter", type=int, default=150,
                   help="LBFGS iterations")
    main(p.parse_args())
