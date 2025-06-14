# src/dataset.py
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from .utils import fetch_ocr

PIXELS = 128
DUP_CAP = 20
STAT = 16

def _label_to_id_scalar(v):
    # direct character conversion 'a' - 'z'
    try:
        vs = str(v).strip().lower()
        if len(vs) == 1 and 'a' <= vs <= 'z':
            return ord(vs) - ord('a')
    except Exception:
        pass

    # numeric forms
    try:
        iv = int(v)
        if 97 <= iv <= 122:  # 'a' to 'z'
            return iv - 97
        if 65 <= iv <= 90:   # 'A' to 'Z'
            return iv - 65
        if 0 <= iv <= 25:    # already 0–25
            return iv
        if 1 <= iv <= 26:    # 1–26 → 0–25
            return iv - 1
    except Exception:
        pass

    # Unrecognized
    return -1

vec_label_to_id = np.vectorize(_label_to_id_scalar, otypes=[int])

class OCRDataset:
    def __init__(self, *, split="train", n_train=2500, window=1, seed=0, mu_sigma=None, dup_cap=DUP_CAP):
        # ---------- load data with correct format ----------
        self.split   = split
        self.window  = window
        self.rng     = np.random.default_rng(seed)

        # ------------------ load raw CSV ------------------
        path = Path(fetch_ocr())
        raw  = pd.read_csv(path, compression="gzip", sep=r"\s+",
                           header=None, comment="#")
        if raw.shape[1] != 134:
            raise ValueError("Unexpected #cols=%d (expect 134)" % raw.shape[1])

        col_names = (['letter_id', 'letter', 'next_id',
                      'word_id', 'position', 'fold'] +
                     [f'stat_{i}'  for i in range(STAT)] +
                     [f'pixel_{i}' for i in range(112)])
        raw.columns = col_names

        stat_cols  = [f'stat_{i}'  for i in range(STAT)]
        pixel_cols = [f'pixel_{i}' for i in range(112)]

        for i in range(len(pixel_cols), PIXELS):
            col = f'pixel_{i}'
            raw[col] = 0.0
            pixel_cols.append(col)

        self.feat_cols = stat_cols + pixel_cols

        raw["label_id"] = vec_label_to_id(raw["letter"].to_numpy())
        raw = raw[raw["label_id"] >= 0]
        
        # ensure each word has only valid labels
        good_words = (raw.groupby("word_id")["label_id"]
                         .apply(lambda x: (x >= 0).all()))
        raw = raw[raw["word_id"].isin(good_words[good_words].index)]

        # ------------------ train/test split ------------------
        spell_per_id = {}                      # word_id → spelling
        for wid, grp in raw.groupby("word_id"):
            spell = ''.join(chr(97 + v) for v in grp["label_id"].to_numpy())
            spell_per_id[wid] = spell

        # group ids by spelling
        ids_by_spell = defaultdict(list)
        for wid, spell in spell_per_id.items():
            ids_by_spell[spell].append(wid)

        spellings = list(ids_by_spell.keys())
        self.rng.shuffle(spellings)

        train_ids, test_ids = [], []
        for sp in spellings:
            ids = ids_by_spell[sp]
            self.rng.shuffle(ids)
            train_ids.extend(ids[:dup_cap])
            test_ids.extend(ids[dup_cap:])

        # trim/pad train_ids to exactly n_train
        if len(train_ids) > n_train:
            self.rng.shuffle(train_ids)
            test_ids.extend(train_ids[n_train:])
            train_ids = train_ids[:n_train]
        elif len(train_ids) < n_train:
            short = n_train - len(train_ids)
            train_ids.extend(test_ids[:short])
            test_ids  = test_ids[short:]

        train_set = set(train_ids)
        mask = raw["word_id"].isin(train_set) if split == "train" \
               else ~raw["word_id"].isin(train_set)

        self.df = raw[mask].copy()

        if split == "train":
            X_full = self.df[self.feat_cols].to_numpy(np.float32)
            self.mu = X_full.mean(axis=0, keepdims=True)
            self.sig = X_full.std(axis=0, keepdims=True)
            self.sig[self.sig == 0] = 1.0
        else:
            if mu_sigma is None:
                raise ValueError("Pass mu_sigma from the train split!")
            self.mu, self.sig = mu_sigma

        # ------------------ build word sequences ------------------
        self.words = []
        pad_len = window // 2
        pad = np.zeros((pad_len, len(self.feat_cols)), dtype=np.float32)

        for wid, grp in self.df.groupby("word_id"):
            grp = grp.sort_values("position")
            X = grp[self.feat_cols].to_numpy(np.float32)
            X = (X - self.mu) / self.sig          # z-score
            y = grp["label_id"].to_numpy(np.int8)

            if len(y) < 2 or len(y) > 20:
                continue

            if window > 1:
                base_dim = len(self.feat_cols)
                X_seq = []
                for i in range(len(y)):
                    neigh = []
                    for d in range(-pad_len, pad_len+1):
                        idx = i + d
                        if 0 <= idx < len(X):
                            neigh.append(X[idx])
                        else:
                            neigh.append(np.zeros(base_dim, dtype=X.dtype))
                    X_seq.append(np.concatenate(neigh, axis=0))
                X = np.vstack(X_seq)
                pos_idx = np.arange(len(y))              # 0,1,2,…,len-1
                pos_clip = np.clip(pos_idx, 0, 9)        # bucket positions ≥9 into the last bin
                pos_feat = np.eye(10, dtype=X.dtype)[pos_clip]   # shape (n,10)

                X = np.hstack([X, pos_feat])             # new dim = old dim + 10
            else:
                pass

            self.words.append({"word_id": wid, "X": X, "y": y})

        # -------------- debug prints ------------------
        print(f"Raw word lengths: {[len(word['y']) for word in self.words[:10]]}")
        print(f"{split.capitalize()} set: {len(self.df)} samples from "
              f"{len(self.words)} words (distinct spellings ≈ {len(ids_by_spell)})")
        if self.words:
            sample = [''.join(chr(97+l) for l in w['y']) for w in self.words[:10]]
            print("Sample words:", sample)

