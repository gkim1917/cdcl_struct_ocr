from pathlib import Path
import numpy as np, argparse, json

def load_history(file_path: Path) -> float:
    """Return the last value (final dev-accuracy) from a 1-D .npy array."""
    arr = np.load(file_path)
    if arr.ndim != 1:
        raise ValueError(f"{file_path} is not a 1-D history array.")
    return float(arr[-1])

def maybe_load_wps(hist_path: Path) -> float | None:
    """Find companion words-per-second file, return its last value or None."""
    stem = hist_path.stem                   # e.g. hist_w3
    # Candidates:   hist_w3.wps.npy    OR    hist_w3_wps.npy
    dot_style   = hist_path.with_suffix(".wps.npy")
    under_style = hist_path.with_name(f"{stem}_wps.npy")

    if dot_style.exists():
        return float(np.load(dot_style)[-1])
    if under_style.exists():
        return float(np.load(under_style)[-1])
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="*.npy files containing dev-accuracy histories")
    args = ap.parse_args()

    results: dict[str, dict[str, float | None]] = {}

    for fname in args.runs:
        hist_p = Path(fname).expanduser()
        if not hist_p.is_file():
            print(f"Skipping (not found): {hist_p}")
            continue

        stem = hist_p.stem
        # Ignore files that are *already* speed logs (stem ends with '_wps')
        if stem.endswith("_wps"):
            continue

        try:
            acc_final = load_history(hist_p)
        except ValueError as e:
            print(f"{e}; skipping.")
            continue

        wps_final = maybe_load_wps(hist_p)

        results[stem] = dict(acc=acc_final, wps=wps_final)

    print(json.dumps(results, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()