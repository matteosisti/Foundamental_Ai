import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.utils.ood_metrics import fpr_at_95_tpr
from src.utils.eomt_post import pixel_probs_from_masks, anomaly_from_pixel_probs, rba_from_masks


def append_metrics_csv(csv_path: Path, row: dict) -> None:
    """Appends a single experiment result row to a global CSV file."""
    write_header = not csv_path.exists()
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Offline Temperature Scaling Sweep for EoMT")
    ap.add_argument("--dataset-name", required=True, help="e.g. RA21")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Directory containing cached logits")
    ap.add_argument("--mode", choices=["robust", "prof-exact"], default="robust")
    ap.add_argument("--method", choices=["msp", "maxlogit", "maxentropy", "rba"], default="msp")
    ap.add_argument("--num-classes", type=int, default=19)
    ap.add_argument("--temperatures", default="0.5,0.75,1.0,1.1,1.25,1.5,2.0", help="Comma-separated T values")
    args = ap.parse_args()

    # Parse temperature list from string
    T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

    root = Path(args.artifacts_dir)
    logits_dir = root / "logits"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # File paths for cached data
    mask_path = logits_dir / f"{args.dataset_name}__eomt_mask_logits_f16.npy"
    class_path = logits_dir / f"{args.dataset_name}__eomt_class_logits_f16.npy"
    gt_path = logits_dir / f"{args.dataset_name}__gt_u8.npy"

    if not mask_path.exists() or not class_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Missing cache files:\n{mask_path}\n{class_path}\n{gt_path}")

    # Load data from disk (CPU-friendly)
    mask_logits = np.load(mask_path)     # [N, Q, H, W]
    class_logits = np.load(class_path)   # [N, Q, C(+1)]
    gt = np.load(gt_path)                # [N, H, W]

    # Define OOD (1) and In-Distribution (0) masks
    ood_mask = (gt == 1)
    ind_mask = (gt == 0)

    if ood_mask.sum() == 0 or ind_mask.sum() == 0:
        raise RuntimeError("GT masks have no OOD or no InD pixels. Check label remapping.")

    # Convert to torch tensors for vectorized math in eomt_post
    mask_t = torch.from_numpy(mask_logits).to(torch.float32)
    class_t = torch.from_numpy(class_logits).to(torch.float32)

    for T in T_list:
        # Calculate anomaly maps based on the selected method
        if args.method == "rba":
            anomaly_t = rba_from_masks(
                mask_logits=mask_t,
                class_logits=class_t,
                num_classes=args.num_classes,
                temperature=float(T),
            )
            anomaly = anomaly_t.numpy()
        else:
            # Composition step for MSP/Entropy/MaxLogit
            pixel_probs = pixel_probs_from_masks(
                mask_logits=mask_t,
                class_logits=class_t,
                num_classes=args.num_classes,
                temperature=float(T),
            )
            anomaly_t = anomaly_from_pixel_probs(pixel_probs, args.method)
            anomaly = anomaly_t.numpy()

        # Extract scores for metrics calculation
        ood_out = anomaly[ood_mask]
        ind_out = anomaly[ind_mask]

        val_out = np.concatenate([ind_out, ood_out])
        val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

        # Compute standard OOD metrics
        auprc = float(average_precision_score(val_label, val_out))
        fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

        metrics = {
            "model": "EoMT",
            "dataset": args.dataset_name,
            "method": args.method,
            "temperature": float(T),
            "mode": args.mode,
            "num_classes": int(args.num_classes),
            "auprc": float(auprc),
            "fpr95": float(fpr95),
            "auprc_pct": float(auprc * 100.0),
            "fpr95_pct": float(fpr95 * 100.0),
            "images_used": int(mask_logits.shape[0]),
            "source": "cache_sweep",
        }

        # Save individual JSON report
        out_json = results_dir / f"{args.dataset_name}__EoMT__{args.method}__T{T}__from_cache.json"
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)

        # Update global summary CSV
        append_metrics_csv(results_dir / "metrics.csv", metrics)

        print(f"[T={T}] Method={args.method} | AUPRC={auprc*100:.2f} | FPR95={fpr95*100:.2f}")


if __name__ == "__main__":
    main()