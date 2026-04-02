"""
scripts/collect_results.py

Reads all metrics.json files from the artifacts directory and
prints a structured summary of all results.

Usage:
    python scripts/collect_results.py --artifacts-dir /path/to/artifacts
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

DATASETS  = ["RA21", "RO21", "LAF", "fs_static", "RA",
             "RA21_1024", "RO21_1024", "LAF_1024", "fs_static_1024", "RA_1024"]
MODELS    = ["ERFNet", "EoMT"]
METHODS   = ["msp", "maxentropy", "maxlogit", "rba"]


def _resolve_timestamp(m: dict, folder_name: str) -> str:
    ts = m.get("timestamp_utc")
    if ts:
        return ts
    # Fallback: extract from folder name "2026-03-31_11-03-26__msp__..."
    return folder_name[:19].replace("_", "T", 1)  # "2026-03-31T11-03-26"


def collect(artifacts_root: str) -> dict:
    results = defaultdict(lambda: defaultdict(dict))
    root = Path(artifacts_root)

    for metrics_path in sorted(root.glob("*/*/*/results/metrics.json")):
        try:
            with open(metrics_path) as f:
                m = json.load(f)
        except Exception as e:
            print(f"[SKIP] {metrics_path}: {e}")
            continue

        dataset = m.get("dataset") or metrics_path.parts[-4]
        model   = m.get("model")   or metrics_path.parts[-3]
        method  = m.get("method")  or "unknown"
        folder  = metrics_path.parts[-3]  # run folder name

        ts_new = _resolve_timestamp(m, folder)

        existing = results[dataset][model].get(method)
        if existing is None:
            results[dataset][model][method] = m
            results[dataset][model][method]["_resolved_ts"] = ts_new
        else:
            ts_old = existing.get("_resolved_ts", "")
            if ts_new > ts_old:
                results[dataset][model][method] = m
                results[dataset][model][method]["_resolved_ts"] = ts_new

    return results

def print_table(results: dict) -> None:
    """Prints a human-readable summary table."""
    for dataset in DATASETS:
        if dataset not in results:
            continue
        print(f"\n{'='*70}")
        print(f"  {dataset}")
        print(f"{'='*70}")
        print(f"  {'Model':<8} {'Method':<14} {'AuPRC':>8} {'FPR95':>8}  {'Images':>6}")
        print(f"  {'-'*50}")
        for model in MODELS:
            if model not in results[dataset]:
                continue
            for method in METHODS:
                m = results[dataset][model].get(method)
                if m is None:
                    continue
                auprc  = m.get("auprc_pct",    m.get("auprc", 0) * 100)
                fpr95  = m.get("fpr95_pct",    m.get("fpr95", 0) * 100)
                n_imgs = m.get("images_used",  "?")
                print(f"  {model:<8} {method:<14} {auprc:>8.2f} {fpr95:>8.2f}  {str(n_imgs):>6}")


def export_json(results: dict, out_path: str) -> None:
    """Exports cleaned results to a single JSON file."""
    clean = {}
    for dataset, models in results.items():
        clean[dataset] = {}
        for model, methods in models.items():
            clean[dataset][model] = {}
            for method, m in methods.items():
                clean[dataset][model][method] = {
                    "auprc":       round(m.get("auprc_pct", m.get("auprc", 0) * 100), 4),
                    "fpr95":       round(m.get("fpr95_pct", m.get("fpr95", 0) * 100), 4),
                    "images_used": m.get("images_used"),
                    "timestamp":   m.get("timestamp_utc"),
                    "resize":      f"{m.get('resize_h','?')}x{m.get('resize_w','?')}",
                    "num_classes": m.get("num_classes"),
                }
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n[SAVED] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", required=True,
                    help="Path to artifacts root directory")
    ap.add_argument("--export-json", default=None,
                    help="Optional: export cleaned results to this JSON file")
    args = ap.parse_args()

    results = collect(args.artifacts_dir)
    print_table(results)

    if args.export_json:
        export_json(results, args.export_json)


if __name__ == "__main__":
    main()