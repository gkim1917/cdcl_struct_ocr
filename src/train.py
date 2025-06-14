# src/train.py
import argparse
import numpy as np
import tqdm
from collections import Counter
from .dataset import OCRDataset
from .models import StructPerceptron

def run(args):
    # Load datasets
    train = OCRDataset(split="train", n_train=args.n_train, window=args.window)
    test  = OCRDataset(split="test",  n_train=args.n_train, window=args.window, mu_sigma=(train.mu, train.sig))
    
    if len(train.words) == 0:
        print("ERROR: Training set is empty!")
        return
    if len(test.words) == 0:
        print("ERROR: Test set is empty!")
        return
    
    # Analyze data distribution
    analyze_data_distribution(train, test)
    
    print(f"Training on {len(train.words)} words")
    print(f"Testing on {len(test.words)} words")
    
    # Initialize model
    model = StructPerceptron(
        feat_dim=args.feat_dim,
        L=26,
        lr=args.lr,
        sat_infer=args.sat_test
    )
    best_acc, best_W = -1, None
    patience_counter = 0
    best_epoch = 0

    # Training loop with improved scheduling
    for epoch in range(args.epochs):
        # More gradual learning rate decay
        if epoch > 0 and epoch % 5 == 0:
            old_lr = model.lr
            model.lr *= 0.8  # gentler decay
            print(f"  LR decay: {old_lr:.4f} -> {model.lr:.4f}")
        
        # Better shuffling - ensure we don't get repeated sequences
        train_words = train.words.copy()
        np.random.shuffle(train_words)
        
        mistakes = char_ok = char_total = 0

        for word in tqdm.tqdm(train_words, desc=f"Epoch {epoch}"):
            X, y_true = word["X"], word["y"].astype(int).tolist()
            
            mistakes += model.update(X, y_true)

            y_pred = model.predict(X)
            char_ok     += sum(p == t for p, t in zip(y_pred, y_true))
            char_total  += len(y_true)
        
        word_acc  = 1 - mistakes / len(train_words)
        char_acc  = char_ok / char_total
        print(f"Epoch {epoch}: word_acc={word_acc:.3f}  char_acc={char_acc:.3f}  lr={model.lr:.4f}")

        # More frequent evaluation for better monitoring
        if epoch % 1 == 0:  # evaluate every epoch
            dev_acc = eval_dev(test, model, limit=1000)  # larger sample
            print(f"  Dev greedy char-acc: {dev_acc:.3f}")
            
            if dev_acc > best_acc:
                best_acc, best_W = dev_acc, (model.W.copy(), model.T.copy())
                best_epoch = epoch
                patience_counter = 0
                print(f"  New best! Saved at epoch {epoch}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Use best weights
    if best_W is not None:
        print(f"Restoring best weights from epoch {best_epoch}")
        model.W, model.T = best_W
    else:
        model.averaged_params()

    # Final evaluation
    if args.sat_test:
        model._sat_enabled = True
    final_acc = eval_dev(test, model)
    print("\n=== FINAL RESULTS ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Dev char-accuracy: {final_acc:.4f}")
    
    # Save results
    np.save("experiments/final_W.npy", model.W)
    np.save("experiments/final_T.npy", model.T)
    
    # Additional analysis
    analyze_predictions(test, model, limit=20)

def analyze_data_distribution(train, test):
    """Analyze and report data distribution issues"""
    print("\n=== DATA ANALYSIS ===")
    
    # Get word spellings
    train_words = [''.join([chr(ord('a') + l) for l in word['y']]) for word in train.words]
    test_words = [''.join([chr(ord('a') + l) for l in word['y']]) for word in test.words]
    
    # Count frequencies
    train_counts = Counter(train_words)
    test_counts = Counter(test_words)
    
    print(f"Training: {len(set(train_words))} unique words from {len(train_words)} samples")
    print(f"Test: {len(set(test_words))} unique words from {len(test_words)} samples")
    
    # Show most frequent words
    print("\nMost frequent training words:")
    for word, count in train_counts.most_common(10):
        print(f"  {word}: {count} ({count/len(train_words)*100:.1f}%)")
    
    print("\nMost frequent test words:")
    for word, count in test_counts.most_common(10):
        print(f"  {word}: {count} ({count/len(test_words)*100:.1f}%)")
    
    # Check for severe imbalance
    max_train_freq = max(train_counts.values()) / len(train_words)
    max_test_freq = max(test_counts.values()) / len(test_words)
    
    if max_train_freq > 0.1:
        print(f"WARNING: Training data severely imbalanced (max freq: {max_train_freq:.1%})")
    if max_test_freq > 0.1:
        print(f"WARNING: Test data severely imbalanced (max freq: {max_test_freq:.1%})")

def analyze_predictions(test, model, limit=20):
    """Analyze model predictions to understand failure modes"""
    print("\n=== PREDICTION ANALYSIS ===")
    
    correct_words = 0
    total_chars_correct = 0
    total_chars = 0
    
    print("Sample predictions (first few errors):")
    errors_shown = 0
    
    for i, word in enumerate(test.words[:limit]):
        if errors_shown >= 10:
            break
            
        pred = model.predict(word["X"])
        truth = word["y"].tolist()
        
        pred_str = ''.join([chr(ord('a') + l) for l in pred])
        truth_str = ''.join([chr(ord('a') + l) for l in truth])
        
        chars_correct = sum(p == t for p, t in zip(pred, truth))
        total_chars_correct += chars_correct
        total_chars += len(truth)
        
        if pred == truth:
            correct_words += 1
        else:
            if errors_shown < 5:
                print(f"  Truth: {truth_str}")
                print(f"  Pred:  {pred_str}")
                print(f"  Chars: {chars_correct}/{len(truth)} correct")
                print()
                errors_shown += 1
    
    print(f"Sample stats: {correct_words}/{limit} words correct ({correct_words/limit:.1%})")
    print(f"Sample chars: {total_chars_correct}/{total_chars} correct ({total_chars_correct/total_chars:.3f})")

def eval_dev(dev_ds, model, limit=None):
    ok = tot = 0
    for i, w in enumerate(dev_ds.words):
        if limit and i >= limit:
            break
        pred   = model.predict(w["X"])
        truth  = w["y"].tolist()
        ok    += sum(p == t for p, t in zip(pred, truth))
        tot   += len(truth)
    return ok / tot

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=2500)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.15)  # slightly lower initial LR
    p.add_argument("--feat_dim", type=int, default=432)
    p.add_argument("--patience", type=int, default=10)  # early stopping patience
    p.add_argument("--sat_test", action="store_true",
               help="use SAT decoding for dev/test")
    run(p.parse_args())