#!/usr/bin/env python
import numpy as np, argparse, json
from src.dataset import OCRDataset
from src.models  import StructPerceptron

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="experiments/final_W.npy")
    ap.add_argument("--n_train", type=int, default=2500)
    args = ap.parse_args()

    # load data
    train = OCRDataset(split="train", n_train=args.n_train, window=3)
    test  = OCRDataset(split="test",  n_train=args.n_train, window=3,
                       mu_sigma=(train.mu, train.sig))

    model = StructPerceptron(feat_dim=train.words[0]["X"].shape[1]-1,
                             sat_train=False)
    model.W = np.load(args.weights)
    model.training = False          # SAT decoding

    preds = []
    for w in test.words:
        preds.append(model.predict(w["X"]))

    json.dump(preds, open("ocr_preds.json", "w"))
    print("Wrote ocr_preds.json")

if __name__ == "__main__":
    main()
