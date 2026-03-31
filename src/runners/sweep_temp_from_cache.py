# src/runners/sweep_temp_from_cache.py
#
# Offline temperature sweep for ERFNet using cached logits.
# No model forward pass needed — loads logits saved by run_erfnet_eval.py.
#
# Expected cache files (written by run_erfnet_eval.py --save-logits):
#   logits/<DATASET>__logits.npy   [N, C, H, W]  float32
#   logits/<DATASET>__gt.npy       [N, H, W]     uint8

import os
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.utils.ood_metrics import fpr_at_95_tpr
from src.utils.determinism import apply_determinism
from src.utils.artifacts import resolve_latest_run_dir_filtered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def append_metrics_csv(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Offline temperature sweep from cached ERFNet logits."
    )

    ap.add_argument("--dataset-name",  required=True, help="e.g. RA21, RO21, LAF")
    ap.add_argument("--artifacts-dir", default="artifacts")

    ap.add_argument("--mode",   choices=["robust", "prof-exact"], required=True)
    ap.add_argument("--method", choices=["msp", "maxentropy", "maxlogit"], required=True)

    ap.add_argument("--temperatures", default="0.5,0.75,1.0,1.1,1.25,1.5,2.0",
                    help="Comma-separated list of temperatures to evaluate")

    ap.add_argument("--use-latest", action="store_true",
                    help="Auto-resolve most recent matching run dir with logits")
    ap.add_argument("--run-dir", default=None,
                    help="Explicit path to a run dir (overrides --use-latest)")

    ap.add_argument("--device",        choices=["cpu", "cuda"], default=None)
    ap.add_argument("--seed",          type=int,  default=0)
    ap.add_argument("--deterministic", action="store_true")

    args = ap.parse_args()

    apply_determinism(mode=args.mode, seed=int(args.seed), deterministic=bool(args.deterministic))

    T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Resolve run directory
    if args.run_dir is not None:
        run_root = Path(os.path.expanduser(args.run_dir))
    else:
        if not args.use_latest:
            raise ValueError("Provide --run-dir or pass --use-latest.")
        ds = args.dataset_name
        run_root = resolve_latest_run_dir_filtered(
            artifacts_root=args.artifacts_dir,
            dataset=ds,
            model="ERFNet",
            method=args.method,
            mode=args.mode,
            require_logits=True,
            logit_files=[f"{ds}__logits.npy", f"{ds}__gt.npy"],
        )

    print(f"[ARTIFACTS] {run_root}")

    # Locate cached logits
    logits_dir = run_root / "logits"
    ds         = args.dataset_name
    l_path     = logits_dir / f"{ds}__logits.npy"
    g_path     = logits_dir / f"{ds}__gt.npy"

    for p in [l_path, g_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing cache file: {p}")

    # Load — move to device ONCE
    logits = np.load(l_path)  # [N, C, H, W] float32
    gt     = np.load(g_path)  # [N, H, W]    uint8

    ood_mask = (gt == 1)
    ind_mask = (gt == 0)
    if ood_mask.sum() == 0 or ind_mask.sum() == 0:
        raise RuntimeError("GT has no OOD or no InD pixels — check remapping / dataset.")

    logits_t = torch.from_numpy(logits).to(device=device, dtype=torch.float32)

    # Output dir
    sweep_dir = run_root / "sweep" / f"{args.method}__{args.mode}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path_out = sweep_dir / "metrics_sweep.csv"

    # Temperature sweep
    for T in T_list:
        Tv = float(T)

        if args.method == "msp":
            p         = F.softmax(logits_t / Tv, dim=1)
            anomaly_t = 1.0 - p.max(dim=1).values

        elif args.method == "maxentropy":
            p         = F.softmax(logits_t / Tv, dim=1)
            anomaly_t = -(p * p.clamp_min(1e-12).log()).sum(dim=1)

        elif args.method == "maxlogit":
            # Temperature has no effect on MaxLogit by definition
            # (operates on raw pre-softmax logits)
            anomaly_t = -logits_t.max(dim=1).values

        else:
            raise ValueError(f"Unknown method: {args.method}")

        anomaly = anomaly_t.detach().cpu().numpy()

        ood_out = anomaly[ood_mask]
        ind_out = anomaly[ind_mask]

        val_out   = np.concatenate([ind_out, ood_out])
        val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

        auprc = float(average_precision_score(val_label, val_out))
        fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

        metrics = {
            "model":         "ERFNet",
            "dataset":       args.dataset_name,
            "method":        args.method,
            "temperature":   Tv,
            "mode":          args.mode,
            "auprc":         auprc,
            "fpr95":         fpr95,
            "auprc_pct":     auprc * 100.0,
            "fpr95_pct":     fpr95 * 100.0,
            "images_used":   int(logits.shape[0]),
            "gt_h":          int(gt.shape[-2]),
            "gt_w":          int(gt.shape[-1]),
            "device":        str(device),
            "seed":          int(args.seed),
            "deterministic": bool(args.deterministic),
            "source":        "logits_cache",
            "run_dir":       str(run_root),
        }

        out_json = sweep_dir / f"T{Tv}__metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        append_metrics_csv(csv_path_out, metrics)
        print(f"[T={Tv}] AUPRC={metrics['auprc_pct']:.4f} | FPR95={metrics['fpr95_pct']:.4f} | saved {out_json}")

    print(f"[DONE] Run used: {run_root}")
    print(f"[DONE] Sweep results in: {sweep_dir}")


if __name__ == "__main__":
    main()
