import os
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.utils.artifacts import resolve_latest_run_dir
from src.utils.ood_metrics import fpr_at_95_tpr


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

	ap.add_argument("--dataset-name", required=True, help="e.g. RA21")
	ap.add_argument("--artifacts-dir", default="artifacts")

	ap.add_argument("--mode", choices=["robust", "prof-exact"], default="robust")

	ap.add_argument("--temperatures", default="0.5,0.75,1.0,1.1,1.25,1.5,2.0", help="comma-separated")

	# selection
	ap.add_argument("--use-latest", action="store_true", help="Use latest ERFNet run automatically")
	ap.add_argument("--run-dir", default=None, help="Optional: manually point to a run directory")

	args = ap.parse_args()

	T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

	if args.run_dir is not None:
		run_root = Path(os.path.expanduser(args.run_dir))
	else:
		if not args.use_latest:
			raise ValueError("Provide --run-dir or use --use-latest")
		run_root = resolve_latest_run_dir(args.artifacts_dir, args.dataset_name, "ERFNet")

	logits_dir = run_root / "logits"
	sweep_dir = run_root / "sweep"
	sweep_dir.mkdir(parents=True, exist_ok=True)

	logits_path = logits_dir / f"{args.dataset_name}__logits.npy"
	gt_path = logits_dir / f"{args.dataset_name}__gt.npy"

	if not logits_path.exists() or not gt_path.exists():
		raise FileNotFoundError(f"Missing cache files: {logits_path} or {gt_path}")

	logits = np.load(logits_path)  # [N,C,H,W]
	gt = np.load(gt_path)          # [N,H,W]

	ood_mask = (gt == 1)
	ind_mask = (gt == 0)

	if ood_mask.sum() == 0 or ind_mask.sum() == 0:
		raise RuntimeError("GT has no OOD or no InD pixels. Check remapping.")

	logits_t = torch.from_numpy(logits)  # float32

	csv_path = sweep_dir / "metrics_sweep.csv"

	for T in T_list:
		p = F.softmax(logits_t / T, dim=1)           # [N,C,H,W]
		msp = torch.max(p, dim=1).values             # [N,H,W]
		anomaly = (1.0 - msp).numpy()                # [N,H,W]

		ood_out = anomaly[ood_mask]
		ind_out = anomaly[ind_mask]

		val_out = np.concatenate([ind_out, ood_out])
		val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

		auprc = float(average_precision_score(val_label, val_out))
		fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

		metrics = {
			"model": "ERFNet",
			"dataset": args.dataset_name,
			"method": "msp",
			"temperature": float(T),
			"mode": args.mode,
			"auprc": auprc,
			"fpr95": fpr95,
			"auprc_pct": auprc * 100.0,
			"fpr95_pct": fpr95 * 100.0,
			"images_used": int(logits.shape[0]),
			"source": "logits_cache",
			"run_dir": str(run_root),
		}

		out_json = sweep_dir / f"T{T}__metrics.json"
		with open(out_json, "w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)

		append_metrics_csv(csv_path, metrics)

		print(f"[T={T}] AUPRC={metrics['auprc_pct']:.4f} | FPR95={metrics['fpr95_pct']:.4f} | saved {out_json}")

	print(f"[DONE] Sweep results in: {sweep_dir}")


if __name__ == "__main__":
	main()