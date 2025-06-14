# tests/sat_eval.py
import argparse, numpy as np, tqdm
from src.dataset import OCRDataset
from src.models   import StructPerceptron

p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--bigrams")
p.add_argument("--n_train", type=int, default=2500)
p.add_argument("--window",  type=int, default=5)
p.add_argument("--feat_dim",type=int, default=730)
args = p.parse_args()

train_ds = OCRDataset(split="train", n_train=args.n_train, window=args.window)
dev = OCRDataset(split="test", n_train=args.n_train, window=args.window, mu_sigma=(train_ds.mu, train_ds.sig))

model = StructPerceptron(feat_dim=args.feat_dim,
                         sat_infer=True)       # SAT on
model.W = np.load(args.weights)
if args.bigrams:
    model.T = np.load(args.bigrams)

ok = tot = 0
for w in tqdm.tqdm(dev.words):
    y = model.predict(w["X"])
    t = w["y"].tolist()
    ok += sum(p==q for p,q in zip(y,t))
    tot+= len(t)
print(f"\nSAT dev char-accuracy: {ok/tot:.4f}")