# merge_results.py
import argparse, json, pickle, csv
from pathlib import Path
import numpy as np

def load_latest_acc(acc_file):
    acc_hist = np.load(acc_file)
    return float(acc_hist[-1])

def load_latest_wps(wps_file):
    if wps_file is None:
        return None
    if Path(wps_file).is_file():
        return float(np.load(wps_file)[-1])
    return None            # CRF has no wps

def main(args):
    registry = json.loads(Path(args.registry).read_text())
    rows = []

    for tag, (acc_file, wps_file) in registry.items():
        acc = load_latest_acc(acc_file)
        wps = load_latest_wps(wps_file)
        rows.append([tag, acc, wps])

    # add CRF (accuracy passed via CLI; wps not applicable)
    rows.append(["CRF", args.crf_acc, None])

    # sort by accuracy descending
    rows.sort(key=lambda r: r[1], reverse=True)

    # -------- Markdown table ----------
    md_lines = ["| Method | Dev char-acc | Words/s |",
                "|:------:|:-----------:|:-------:|"]
    for tag, acc, wps in rows:
        md_lines.append(f"| **{tag}** | {acc:.4f} | "
                        f"{wps:.0f} |" if wps is not None else
                        f"| **{tag}** | {acc:.4f} | — |")
    md_table = "\n".join(md_lines)
    Path(args.md_out).write_text(md_table)
    print(f"Markdown table written → {args.md_out}")

    # -------- CSV (optional) ----------
    with Path(args.csv_out).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dev_char_acc", "words_per_sec"])
        writer.writerows(rows)
    print(f"CSV written → {args.csv_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="results_register.json",
                   help="JSON file mapping tag → [acc_npy, wps_npy]")
    p.add_argument("--crf_acc", type=float, required=True,
                   help="Dev accuracy you saw for CRF (printed after training)")
    p.add_argument("--md_out",  default="docs/results_table.md")
    p.add_argument("--csv_out", default="docs/results_table.csv")
    main(p.parse_args())
