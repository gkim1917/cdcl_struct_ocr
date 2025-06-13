import numpy as np
import pandas as pd
from pathlib import Path
from .utils import fetch_ocr

class OCRDataset:
    def __init__(self, split="train", n_train=2500, window=3, seed=0):
        # ---------- 1. load ----------
        path = Path(fetch_ocr())
        raw = pd.read_csv(
            path,
            compression="gzip",
            sep=r"\s+",          # use regex instead of deprecated delim_whitespace
            header=None,
            comment="#",
        )

        # ---------- 2. infer layout ----------
        n_total_cols = raw.shape[1]          # 134
        PIXELS = 128
        pixel_start = n_total_cols - PIXELS  # 6 meta columns
        meta = raw.iloc[:, :pixel_start]
        pixels = raw.iloc[:, pixel_start:]

        # meta columns 0-2 → word_id, position, label-char
        raw = pd.concat({"word_id": meta.iloc[:, 0],
                         "position": meta.iloc[:, 1],
                         "char": meta.iloc[:, 2],
                         **{i: pixels.iloc[:, i] for i in range(PIXELS)}},
                        axis=1)

        # ---------- 3. train/test split ----------
        rng = np.random.default_rng(seed)
        word_ids = raw["word_id"].unique()
        rng.shuffle(word_ids)
        train_ids = set(word_ids[:n_train])
        mask = raw["word_id"].isin(train_ids) if split == "train" else ~raw["word_id"].isin(train_ids)
        self.df = raw[mask]
        self.window = window
        self.build_sequences()

    def build_sequences(self):
        self.words = []
        for wid, grp in self.df.groupby("word_id"):
            grp = grp.sort_values("position")
            X = grp.iloc[:, 3:].to_numpy(dtype=np.float32) / 255.0  # pixels only
            y = grp["char"].to_numpy()
            if self.window > 1:
                pad = np.zeros((self.window // 2, X.shape[1]), dtype=X.dtype)
                X = np.vstack([pad, X, pad])
            self.words.append({"X": X, "y": y})
