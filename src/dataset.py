# src/dataset.py
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from .utils import fetch_ocr

PIXELS = 128

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i for i, c in enumerate(ALPHABET)}          # 'A'→0 … 'Z'→25
CHAR2IDX.update({c.lower(): i for c, i in CHAR2IDX.items()})

def _label_to_id_scalar(v):
    """
    Map a single OCR label to integer 0-25.

    • Accepts: single char 'A'..'Z'/'a'..'z', ASCII code, or already-numeric 0-25.
    • Returns: int in [0, 25]  OR  -1 if the label is unrecognised.
    """
    # 1️⃣  Already numeric?
    try:
        iv = int(v)
        if 0 <= iv <= 25:
            return iv
        # Some datasets store ASCII code (e.g. 65 = 'A')
        if 65 <= iv <= 90:
            return iv - 65
        if 97 <= iv <= 122:
            return iv - 97
    except (ValueError, TypeError):
        pass

    # 2️⃣  Single-character string?
    try:
        vs = str(v).strip()  # Remove whitespace
        if len(vs) == 1 and vs in CHAR2IDX:
            return CHAR2IDX[vs]
    except:
        pass

    # 3️⃣  Otherwise unrecognised
    return -1


def _labels_to_ids(arr):
    """Convert array of labels to IDs, filtering out unknown labels"""
    vec_map = np.vectorize(_label_to_id_scalar, otypes=[int])
    ids = vec_map(arr)
    return ids

class OCRDataset:
    def __init__(self, split="train", n_train=2500, window=3, seed=0):
        # ---------- 1. load & name columns ----------
        path = Path(fetch_ocr())
        n_cols = 6 + PIXELS                      # 6 meta + 128 pixels = 134
        names = (
            ["word_id", "position", "label", "fold", "x_box", "y_box"]
            + [f"p{i}" for i in range(PIXELS)]
        )
        raw = pd.read_csv(
            path,
            compression="gzip",
            sep=r"\s+",           # whitespace-split
            header=None,
            names=names,
            comment="#",
        )

        # ---------- 2. normalise labels -------------
        raw["label_id"] = _labels_to_ids(raw["label"].to_numpy())

        # Drop rows whose label_id == -1 (unrecognised) AND ensure we keep word integrity
        valid_mask = raw["label_id"] >= 0
        
        # Get word_ids that have ALL valid labels
        word_validity = raw.groupby("word_id")["label_id"].apply(lambda x: (x >= 0).all())
        valid_word_ids = word_validity[word_validity].index
        
        # Keep only complete valid words
        raw = raw[raw["word_id"].isin(valid_word_ids)]
        
        n_dropped = len(raw) - valid_mask.sum()
        if n_dropped > 0:
            warnings.warn(f"Dropped {n_dropped} rows with unknown labels or incomplete words.")

        # ---------- 3. split train / test -----------
        rng = np.random.default_rng(seed)
        word_ids = raw["word_id"].unique()
        rng.shuffle(word_ids)
        
        # Ensure we have enough words for both train and test
        if len(word_ids) < n_train:
            warnings.warn(f"Only {len(word_ids)} words available, requested {n_train} for training.")
            n_train = min(n_train, len(word_ids) // 2)  # Use half for training if not enough
        
        train_set = set(word_ids[:n_train])
        mask = raw["word_id"].isin(train_set) if split == "train" else ~raw["word_id"].isin(train_set)
        self.df = raw.loc[mask]
        
        print(f"{split.capitalize()} set: {len(self.df)} samples from {len(self.df['word_id'].unique())} words")

        self.window = window
        self.pixel_cols = [f"p{i}" for i in range(PIXELS)]
        self.build_sequences()

    def build_sequences(self):
        self.words = []
        for _, grp in self.df.groupby("word_id"):
            grp = grp.sort_values("position")
            X = grp[self.pixel_cols].to_numpy(dtype=np.float32) / 255.0
            y = grp["label_id"].to_numpy(dtype=np.int8)

            # Verify all labels are valid
            if np.any(y < 0) or np.any(y > 25):
                warnings.warn(f"Invalid labels found in word {grp['word_id'].iloc[0]}: {y}")
                continue

            if self.window > 1:
                pad = np.zeros((self.window // 2, PIXELS), dtype=X.dtype)
                X = np.vstack([pad, X, pad])

            self.words.append({"X": X, "y": y})