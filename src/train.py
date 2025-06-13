# src/train.py
import argparse, numpy as np, tqdm
from .dataset import OCRDataset
from .models import StructPerceptron

def run(args):
    train = OCRDataset(split="train", n_train=args.n_train, window=args.window)
    test  = OCRDataset(split="test",  n_train=args.n_train, window=args.window)
    
    # Check if datasets are not empty
    if len(train.words) == 0:
        print("ERROR: Training set is empty!")
        return
    if len(test.words) == 0:
        print("ERROR: Test set is empty!")
        return
    
    model = StructPerceptron(lr=args.lr)
    
    for epoch in range(args.epochs):
        mistakes = 0
        for word in tqdm.tqdm(train.words, desc=f"Epoch {epoch}"):
            y_true = word["y"].astype(int).tolist()
            
            # Validate labels before training
            if any(y < 0 or y > 25 for y in y_true):
                print(f"Warning: Invalid labels found: {y_true}")
                continue
                
            mistakes += model.update(word["X"], y_true)
        print(f"Epoch {epoch}: mistakes={mistakes}")
    
    # evaluation
    correct, total = 0, 0
    for word in test.words:
        y_pred = model.predict(word["X"])
        y_true = word["y"].astype(int).tolist()
        
        # Ensure predictions and true labels have same length
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
        
        correct += sum(p == t for p, t in zip(y_pred, y_true))
        total += len(y_true)
    
    if total > 0:
        print("Test accuracy:", correct/total)
    else:
        print("No test samples available for evaluation!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1.0)
    run(p.parse_args())