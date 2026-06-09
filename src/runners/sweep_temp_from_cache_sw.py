# src/runners/sweep_temp_from_cache_sw.py
#
# Offline temperature sweep for EoMT sliding window mode.
# Uses cached pixel logits [N, C, H, W] saved by run_eomt_eval.py
# with --sliding-window --save-logits.
#
# Unlike sweep_temp_from_cache_eomt.py which works on raw mask/class logits,
# this script works on pre-recomposed pixel logits — one tensor per image
# at original resolution. This is the output of the SlidingWindow pipeline.
#
# Supported methods: msp, maxentropy, maxlogit
# Note: RbA is not supported (requires per-query decomposition).

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


def append_metrics_csv(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def anomaly_from_pixel_logits_t(
    pixel_logits: torch.Tensor,
    method: str,
    T: float,
) -> torch.Tensor:
    """
    Computes anomaly score from pixel logits [N, C, H, W] at temperature T.

    Returns anomaly map [N, H, W].
    """
    if method == "maxlogit":
        # Temperature-invariant
        return (1.0 - pixel_logits.max(dim=1).values)

    scaled = pixel_logits / T

    if method in ("msp", "rba"):
        probs = scaled.softmax(dim=1)
        return (1.0 - probs.max(dim=1).values)

    if method == "maxentropy":
        probs = scaled.softmax(dim=1)
        return -(probs * (probs + 1e-8).log()).sum(dim=1)

    raise ValueError(f"Unknown method: {method}")


def main():
    ap = argparse.ArgumentParser(
        description="Offline temperature sweep from cached SW pixel logits."
    )
    ap.add_argument("--dataset-name",  required=True)
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--method",        choices=["msp", "maxentropy", "maxlogit", "rba"],
                    default="msp")
    ap.add_argument("--mode",          choices=["robust", "prof-exact"], default="robust")
    ap.add_argument("--temperatures",  default="0.5,0.75,1.0,1.1,1.25,1.5,2.0",
                    help="Comma-separated temperature values")

    ap.add_argument("--use-latest",    action="store_true")
    ap.add_argument("--run-dir",       default=None)

    ap.add_argument("--device",        choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")

    args = ap.parse_args()

    if args.method == "rba":
        return -torch.tanh(pixel_logits).sum(dim=1)

    apply_determinism(mode=args.mode, seed=args.seed, deterministic=args.deterministic)

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

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
            model="EoMT",
            method=args.method,
            mode=args.mode,
            require_logits=True,
            logit_files=[f"{ds}__pixel_logits_f16.npy", f"{ds}__gt.npy"],
        )

    print(f"[ARTIFACTS] {run_root}")

    ds          = args.dataset_name
    logits_dir  = run_root / "logits"
    pl_path     = logits_dir / f"{ds}__pixel_logits_f16.npy"
    gt_path     = logits_dir / f"{ds}__gt.npy"

    for p in [pl_path, gt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing cache file: {p}")

    # Load — pixel logits [N, C, H, W] float16 → float32
    pixel_logits = torch.from_numpy(
        np.load(pl_path).astype(np.float32)
    ).to(device)   # [N, C, H, W]

    gt = np.load(gt_path)  # [N, H, W] uint8

    sweep_dir = run_root / "sweep" / f"{args.method}__sw__{args.mode}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows = []

    for Tv in T_list:
        anomaly = anomaly_from_pixel_logits_t(
            pixel_logits, method=args.method, T=Tv
        ).cpu().numpy()   # [N, H, W]

        ood_out = anomaly[gt == 1]
        in_out  = anomaly[gt == 0]

        val_out   = np.concatenate([in_out, ood_out])
        val_label = np.concatenate([np.zeros(len(in_out)), np.ones(len(ood_out))])

        auprc = float(average_precision_score(val_label, val_out))
        fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

        row = {
            "T":           Tv,
            "method":      args.method,
            "mode":        args.mode,
            "sw":          True,
            "auprc":       auprc,
            "fpr95":       fpr95,
            "auprc_pct":   auprc * 100.0,
            "fpr95_pct":   fpr95 * 100.0,
        }

        t_str    = str(Tv).replace(".", "p")
        out_path = sweep_dir / f"T{t_str}__metrics.json"
        with open(out_path, "w") as f:
            json.dump(row, f, indent=2)

        sweep_rows.append(row)
        print(f"[T={Tv}] AUPRC={auprc*100:.4f} | FPR95={fpr95*100:.4f} | saved {out_path}")

    csv_path = sweep_dir / "metrics_sweep.csv"
    for row in sweep_rows:
        append_metrics_csv(csv_path, row)

    print(f"[DONE] Sweep results in: {sweep_dir}")


if __name__ == "__main__":
    main()
