# =========================
# File: scripts/erfnet_temp_sweep_from_cache.py
# (riscrittura completa del tuo secondo file)
# =========================
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

# Robust import
try:
	from src.utils.ood_metrics import fpr_at_95_tpr
except Exception:
	from ood_metrics import fpr_at_95_tpr


def append_metrics_csv(csv_path: Path, row: dict) -> None:
	write_header = not csv_path.exists()
	fieldnames = list(row.keys())
	with open(csv_path, "a", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		if write_header:
			writer.writeheader()
		writer.writerow(row)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dataset-name", required=True, help="es: RA21")
	ap.add_argument("--artifacts-dir", default="artifacts")
	ap.add_argument("--temperatures", default="0.5,0.75,1.0,1.1,1.25,1.5,2.0", help="comma-separated")
	args = ap.parse_args()

	T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

	root = Path(args.artifacts_dir)
	logits_dir = root / "logits"
	results_dir = root / "results"
	results_dir.mkdir(parents=True, exist_ok=True)

	logits_path = logits_dir / f"{args.dataset_name}__logits.npy"
	gt_path = logits_dir / f"{args.dataset_name}__gt.npy"

	if not logits_path.exists() or not gt_path.exists():
		raise FileNotFoundError(f"Missing cache files: {logits_path} or {gt_path}")

	logits = np.load(logits_path)  # [N,C,H,W]
	gt = np.load(gt_path)		  # [N,H,W]

	ood_mask = (gt == 1)
	ind_mask = (gt == 0)

	if ood_mask.sum() == 0 or ind_mask.sum() == 0:
		raise RuntimeError("GT masks have no OOD or no InD pixels. Check dataset remapping.")

	logits_t = torch.from_numpy(logits)  # float32

	for T in T_list:
		# MSP(T): 1 - max softmax(logits/T)
		p = F.softmax(logits_t / T, dim=1)				 # [N,C,H,W]
		msp = torch.max(p, dim=1).values				 # [N,H,W]
		anomaly = (1.0 - msp).numpy()					 # [N,H,W]

		ood_out = anomaly[ood_mask]
		ind_out = anomaly[ind_mask]

		val_out = np.concatenate([ind_out, ood_out])
		val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

		auprc = average_precision_score(val_label, val_out)
		fpr95 = fpr_at_95_tpr(val_out, val_label)

		metrics = {
			"model": "ERFNet",
			"dataset": args.dataset_name,
			"method": "msp",
			"temperature": float(T),
			"auprc": float(auprc),
			"fpr95": float(fpr95),
			"auprc_pct": float(auprc * 100.0),
			"fpr95_pct": float(fpr95 * 100.0),
			"images_used": int(logits.shape[0]),
			"source": "logits_cache",
		}

		out_json = results_dir / f"{args.dataset_name}__ERFNet__msp__T{T}__from_cache.json"
		with open(out_json, "w") as f:
			json.dump(metrics, f, indent=2)

		append_metrics_csv(results_dir / "metrics.csv", metrics)

		print(f"[T={T}] AUPRC={auprc*100:.4f} | FPR95={fpr95*100:.4f} | saved {out_json}")


if __name__ == "__main__":
	main()