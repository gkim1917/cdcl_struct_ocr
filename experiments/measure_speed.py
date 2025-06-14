#  experiments/measure_speed.py
import argparse, time, numpy as np
from pathlib import Path
from src.dataset import OCRDataset
from src.models  import StructPerceptron

ap = argparse.ArgumentParser()
ap.add_argument("--weights",   default="experiments/final_W.npy")
ap.add_argument("--bigrams",   default="experiments/final_T.npy")
ap.add_argument("--window",    type=int, required=True)
ap.add_argument("--feat_dim",  type=int, required=False)
ap.add_argument("--n_train",   type=int, default=2500)
ap.add_argument("--tag",       type=str,  required=True,
                help="same tag as the hist_*.npy file (e.g. w3, w7 ...)")
args = ap.parse_args()

def load_model(weights_p: Path, bigrams_p: Path, feat_dim: int) -> StructPerceptron:
    W = np.load(weights_p)
    T = np.load(bigrams_p) if bigrams_p.is_file() else np.zeros((26, 26), np.float32)

    feat_dim = W.shape[1] - 1 
    model = StructPerceptron(feat_dim=feat_dim)
    model.W, model.T = W.astype(np.float32), T.astype(np.float32)
    return model, feat_dim


def main() -> None:
    args = ap.parse_args()

    weights_p  = Path(args.weights).expanduser()
    bigrams_p  = Path(args.bigrams).expanduser()
    out_path   = Path(f"experiments/wps_hist_{args.tag}.npy")

    model, feat_dim = load_model(weights_p, bigrams_p, args.feat_dim)

    # ---------------- build minibatch dev set ---------------------------
    train_ds = OCRDataset(split="train",
                          n_train=args.n_train,
                          window=args.window)
    dev_ds   = OCRDataset(split="test",
                          n_train=args.n_train,
                          window=args.window,
                          mu_sigma=(train_ds.mu, train_ds.sig))

    words = dev_ds.words[:500]        # fixed probe size
    start = time.perf_counter()
    for w in words:
        _ = model.predict(w["X"])
    duration = time.perf_counter() - start
    wps = len(words) / duration
    print(f"{args.tag}: {wps:.1f} words / s (greedy)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.array([wps], dtype=np.float32))
    print(f"Saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()