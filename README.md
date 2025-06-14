# CDCL Structured OCR/POS

### Overview
This repository contains an end‑to‑end pipeline for SAT‑based structured prediction on the classic OCR letter‑sequence benchmark.  We train a unary–bigram structured perceptron and decode either with

- Greedy / Viterbi (fast, baseline)

- CNF + Max‑SAT decoding (exact, supports hard/soft constraints)

Recent additions (May 2025) include:

- Pruned SAT decoder (src/sat_infer.py) — skips low‑impact transitions and low‑probability labels for 10‑200× speed‑ups.

- Position‑window features (window = 5, feat_dim = 730)

- Data‑imbalance handling via DUP_CAP and z‑score normalisation.

## Dev accuracy (window 5, n_train 2500):
       decoder       |      char‑accuracy
---------------------------------------------
        Greedy       |       ≈ 0.715
     SAT (pruned)    |       ≈ 0.713 (2 s)


## Installation
Tested on Python 3.10 + macOS 13: 
```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Repo Structure
.
├── src/
│   ├── dataset.py        # data loader & normalisation
│   ├── models.py         # structured perceptron
│   ├── cnf_encoder.py    # ultra‑light CNF / Max‑SAT encoder
│   ├── sat_infer.py      # SAT/Max‑SAT decoder
│   └── train.py          # training script
├── tests/                # unit & dev‑set evaluators
├── submission/
│   └── checkpoints/      # saved weights for reproduction
└── requirements.txt


## Usage 
### 1. Train
To reproduces the model whose weights are already stored in submission/checkpoints/final_W.npy & final_T.npy:
```
PYTHONPATH=$PWD python -m src.train \
    --n_train 2500 --window 5 --feat_dim 730 \
    --epochs 25 --lr 0.15
```
### 2. Greedy / Viterbi evaluation
To run greedy/viterbi evaluation:
```
PYTHONPATH=$PWD python tests/greedy_eval.py \
    --weights submission/checkpoints/final_W.npy \
    --n_train 2500 --window 5 --feat_dim 730
```
### 3. SAT evaluation
To run sat evaluation:
```
PYTHONPATH=$PWD python tests/sat_eval.py \
    --weights submission/checkpoints/final_W.npy \
    --bigrams submission/checkpoints/final_T.npy \
    --n_train 2500 --window 5 --feat_dim 730
```

## Algorithm details
- Unary features — 112 image pixels + 16 handcrafted statistics per letter, concatenated over a context window (window×feat_dim).  A bias term is appended automatically.

- Bigram matrix T — learned by perceptron updates; rewards frequent letter pairs.

- SAT formulationVariables v_{i,l} (choose label l at pos i) and optional z_{i,l,q} (choose bigram l→q)
- - Hard clauses: exactly‑one per position; linking z → v.
- - Soft clauses: maximise unaries & bigram rewards (converted to costs for RC2).
- - Pruning: skip transitions T[l,q] < T_max − τ (τ = 2.0 by default).

Implementation lives in src/sat_infer.py and passes tests/test_sat.py.


## License

MIT License — see LICENSE file.

## Contact

G. Kim  ·  gkim@ucsd.edu   ·  COGS 185 Spring 2025


