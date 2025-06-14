#tests/greedy_eval.py
import argparse
import numpy as np
import tqdm

from src.dataset import OCRDataset
from src.models  import StructPerceptron


def evaluate(dev_ds, model):
    """Return character-level accuracy on *dev_ds* using greedy decoding."""
    correct, total = 0, 0
    for word in tqdm.tqdm(dev_ds.words, desc="dev", ncols=80):
        y_pred  = model.predict(word["X"])        # greedy / Viterbi in model
        y_true  = word["y"].tolist()
        correct += sum(p == t for p, t in zip(y_pred, y_true))
        total   += len(y_true)
    return correct / total if total else 0.0


def main(args):
    # ---------- data ----------
    train = OCRDataset(split="train",
                       n_train=args.n_train,
                       window=args.window)
    dev   = OCRDataset(split="test",
                       n_train=args.n_train,
                       window=args.window,
                       mu_sigma=(train.mu, train.sig))

    # ---------- model ----------
    model = StructPerceptron(feat_dim=args.feat_dim, L=26, lr=0.0)   # lr not used here
    model.W = np.load(args.weights)
    if args.bigrams:
        model.T = np.load(args.bigrams)

    # ---------- evaluation ----------
    acc = evaluate(dev, model)
    print(f"\nGreedy dev char-accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="Path to saved unary weight matrix (Numpy .npy)")
    parser.add_argument("--bigrams", default=None,
                        help="Optional path to saved bigram matrix (.npy)")
    parser.add_argument("--n_train", type=int, default=2500,
                        help="#training words used to build µ/σ stats")
    parser.add_argument("--window",  type=int, default=3,
                        help="Context window size used when saving the model")
    parser.add_argument("--feat_dim", type=int, default=432,
                        help="Feature dimension per time step *before* window concat")

    main(parser.parse_args())
