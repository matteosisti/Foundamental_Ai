# src/runners/sweep_temp_from_cache.py
import os
import re
import json
import csv
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.utils.ood_metrics import fpr_at_95_tpr

# Regex to match the new 'pretty' run directory naming convention
RUN_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})__"
    r"(?P<method>[^_]+)__"
    r"T(?P<T>[^_]+)__"
    r"(?P<mode>robust|prof-exact)__"
    r"(?P<hash>[0-9a-fA-F]+)$"
)

def append_metrics_csv(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def parse_run_dir_name(name: str) -> Optional[dict]:
    """Parses run directory name to extract metadata via regex."""
    m = RUN_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "ts": d["ts"],
        "method": d["method"].lower(),
        "T": d["T"],
        "mode": d["mode"].lower(),
        "hash": d["hash"].lower(),
    }

def find_latest_run_dir(
    artifacts_root: str,
    dataset: str,
    model: str,
    method: str,
    mode: str,
    require_logits: bool = True,
) -> Path:
    """Discovers the most recent run matching the specified method and mode."""
    root = Path(os.path.expanduser(artifacts_root))
    base = root / dataset / model
    if not base.exists():
        raise FileNotFoundError(f"Base artifacts folder not found: {base}")

    method = method.lower().strip()
    mode = mode.lower().strip()

    candidates: List[Tuple[str, Path]] = []

    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue

        info = parse_run_dir_name(run_dir.name)
        if info is None:
            continue

        if info["method"] != method or info["mode"] != mode:
            continue

        if require_logits:
            logits_dir = run_dir / "logits"
            # Standard cached filenames from eval_erfnet.py
            l_path = logits_dir / f"{dataset}__logits.npy"
            g_path = logits_dir / f"{dataset}__gt.npy"
            if not l_path.exists() or not g_path.exists():
                continue

        candidates.append((info["ts"], run_dir))

    if not candidates:
        raise FileNotFoundError(f"No matching runs found in {base} for {method}/{mode}")

    # Sort by timestamp descending to get the latest
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def main():
    ap = argparse.ArgumentParser(description="Temperature sweep from cached logits.")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--mode", choices=["robust", "prof-exact"], required=True)
    ap.add_argument("--method", choices=["msp", "maxentropy", "maxlogit"], required=True)
    ap.add_argument("--temperatures", default="0.5,0.75,1.0,1.1,1.25,1.5,2.0")
    ap.add_argument("--use-latest", action="store_true")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

    # Resolve run directory
    if args.run_dir:
        run_root = Path(os.path.expanduser(args.run_dir))
    else:
        if not args.use_latest:
            raise ValueError("Provide --run-dir or use --use-latest")
        run_root = find_latest_run_dir(
            args.artifacts_dir, args.dataset_name, "ERFNet", args.method, args.mode
        )

    logits_dir = run_root / "logits"
    sweep_dir = run_root / "sweep" / f"{args.method}__{args.mode}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Load cache
    l_path = logits_dir / f"{args.dataset_name}__logits.npy"
    g_path = logits_dir / f"{args.dataset_name}__gt.npy"
    
    logits = np.load(l_path)  # [N,C,H,W]
    gt = np.load(g_path)      # [N,H,W]

    ood_mask = (gt == 1)
    ind_mask = (gt == 0)

    # Convert to torch once for GPU/CPU efficiency
    logits_t = torch.from_numpy(logits)
    csv_path = sweep_dir / "metrics_sweep.csv"

    for T in T_list:
        if args.method == "msp":
            p = F.softmax(logits_t / T, dim=1)
            anomaly = (1.0 - torch.max(p, dim=1).values).numpy()

        elif args.method == "maxentropy":
            p = F.softmax(logits_t / T, dim=1)
            anomaly = (-(p * torch.clamp_min(p, 1e-12).log()).sum(dim=1)).numpy()

        elif args.method == "maxlogit":
            # Temperature scaling on maxlogit is monotonic but included for completeness
            anomaly = (-torch.max(logits_t / T, dim=1).values).numpy()

        ood_out = anomaly[ood_mask]
        ind_out = anomaly[ind_mask]

        val_out = np.concatenate([ind_out, ood_out])
        val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

        auprc = float(average_precision_score(val_label, val_out))
        fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

        res = {
            "model": "ERFNet",
            "dataset": args.dataset_name,
            "method": args.method,
            "temperature": T,
            "mode": args.mode,
            "auprc_pct": auprc * 100.0,
            "fpr95_pct": fpr95 * 100.0,
            "run_dir": str(run_root)
        }

        with open(sweep_dir / f"T{T}__metrics.json", "w") as f:
            json.dump(res, f, indent=2)
        append_metrics_csv(csv_path, res)

        print(f"[T={T}] AUPRC={res['auprc_pct']:.4f} | FPR95={res['fpr95_pct']:.4f}")

    print(f"\n[DONE] Results in: {sweep_dir}")

if __name__ == "__main__":
    main()