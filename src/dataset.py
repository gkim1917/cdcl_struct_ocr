# src/dataset.py
import numpy as np, pandas as pd
from pathlib import Path
from .utils import fetch_ocr

COLS = ["word_id","position","char","x-box","y-box","width","height","pixels"]  # truncated

import pandas as pd
from pathlib import Path
from .utils import fetch_ocr

def _load_raw():
    path = Path(fetch_ocr())
    raw = pd.read_csv(
        path,
        compression="gzip",
        delim_whitespace=True,
        header=None,
        comment="#",
    )
    return raw

class OCRDataset:
    def __init__(self, split="train", n_train=2500, window=3, seed=0):
        raw = _load_raw()
        raw.columns = COLS + list(range(128))  # 16×8 =128 pixels
        rng = np.random.default_rng(seed)
        word_ids = raw["word_id"].unique()
        rng.shuffle(word_ids)
        train_ids = set(word_ids[:n_train])
        mask = raw["word_id"].isin(train_ids) if split=="train" else ~raw["word_id"].isin(train_ids)
        self.df = raw[mask]
        self.window = window
        self.build_sequences()

    def build_sequences(self):
        self.words = []
        for wid, group in self.df.groupby("word_id"):
            chars = group.sort_values("position")
            X = chars.iloc[:, -128:].to_numpy() / 255.0
            y = chars["char"].to_numpy()
            # pad windows on both sides
            if self.window>1:
                pad = np.zeros((self.window//2, 128))
                X = np.vstack([pad, X, pad])
            self.words.append({"X": X, "y": y})
