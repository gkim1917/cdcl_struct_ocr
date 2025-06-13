# src/train.py
import argparse, numpy as np, tqdm
from .dataset import OCRDataset
from .models import StructPerceptron

def run(args):
    train = OCRDataset(split="train", n_train=args.n_train, window=args.window)
    test  = OCRDataset(split="test",  n_train=args.n_train, window=args.window)
    model = StructPerceptron(lr=args.lr)
    for epoch in range(args.epochs):
        mistakes = 0
        for word in tqdm.tqdm(train.words, desc=f"Epoch {epoch}"):
            mistakes += model.update(word["X"], [ord(c)-97 for c in word["y"]])
        print(f"Epoch {epoch}: mistakes={mistakes}")
    # evaluation
    correct, total = 0,0
    for word in test.words:
        y_pred = model.predict(word["X"])
        correct += sum(p==ord(c)-97 for p,c in zip(y_pred, word["y"]))
        total   += len(word["y"])
    print("Test accuracy:", correct/total)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1.0)
    run(p.parse_args())
