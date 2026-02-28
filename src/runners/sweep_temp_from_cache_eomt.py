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


def pixel_probs_from_masks(mask_logits: torch.Tensor, class_logits: torch.Tensor, num_classes: int, temperature: float) -> torch.Tensor:
	# mask_logits: [N,Q,h,w], class_logits: [N,Q,C(+1)]
	if class_logits.shape[-1] == num_classes + 1:
		class_logits = class_logits[..., :num_classes]

	class_prob = F.softmax(class_logits / temperature, dim=-1)	# [N,Q,C]
	mask_prob = torch.sigmoid(mask_logits)						# [N,Q,h,w]

	pixel = torch.einsum("nqc,nqhw->nchw", class_prob, mask_prob)	# [N,C,h,w]
	den = pixel.sum(dim=1, keepdim=True).clamp_min(1e-8)
	pixel = pixel / den
	return pixel


def anomaly_from_pixel_probs(pixel_probs: torch.Tensor, method: str) -> torch.Tensor:
	if method == "msp":
		msp = pixel_probs.max(dim=1).values
		return 1.0 - msp

	if method == "maxentropy":
		ent = -(pixel_probs * pixel_probs.clamp_min(1e-12).log()).sum(dim=1)
		return ent

	if method == "maxlogit":
		logits_proxy = pixel_probs.clamp_min(1e-12).log()
		m = logits_proxy.max(dim=1).values
		return -m

	raise ValueError(f"Unknown method: {method}")


def rba_from_masks(mask_logits: torch.Tensor, class_logits: torch.Tensor, num_classes: int, temperature: float, area_pow: float = 0.5) -> torch.Tensor:
	if class_logits.shape[-1] == num_classes + 1:
		class_logits = class_logits[..., :num_classes]

	class_prob = F.softmax(class_logits / temperature, dim=-1)	# [N,Q,C]
	conf = class_prob.max(dim=-1).values						# [N,Q]

	mask_prob = torch.sigmoid(mask_logits)						# [N,Q,h,w]
	area = mask_prob.mean(dim=(-2, -1))						# [N,Q]

	reliability = conf * (area.clamp_min(1e-6) ** area_pow)		# [N,Q]
	reliability = reliability.unsqueeze(-1).unsqueeze(-1)		# [N,Q,1,1]

	normality = (reliability * mask_prob).amax(dim=1)			# [N,h,w]
	anomaly = 1.0 - normality
	return anomaly


def main():
	ap = argparse.ArgumentParser()

	ap.add_argument("--dataset-name", required=True, help="e.g. RA21")
	ap.add_argument("--artifacts-dir", default="artifacts")

	ap.add_argument("--mode", choices=["robust", "prof-exact"], default="robust")
	ap.add_argument("--method", choices=["msp", "maxentropy", "maxlogit", "rba"], default="msp")

	ap.add_argument("--num-classes", type=int, default=19)
	ap.add_argument("--temperatures", default="0.5,0.75,1.0,1.1,1.25,1.5,2.0", help="comma-separated")

	ap.add_argument("--use-latest", action="store_true")
	ap.add_argument("--run-dir", default=None)

	args = ap.parse_args()

	T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

	if args.run_dir is not None:
		run_root = Path(os.path.expanduser(args.run_dir))
	else:
		if not args.use_latest:
			raise ValueError("Provide --run-dir or use --use-latest")
		run_root = resolve_latest_run_dir(args.artifacts_dir, args.dataset_name, "EoMT")

	logits_dir = run_root / "logits"
	sweep_dir = run_root / "sweep"
	sweep_dir.mkdir(parents=True, exist_ok=True)

	mask_path = logits_dir / f"{args.dataset_name}__mask_logits_f16.npy"
	class_path = logits_dir / f"{args.dataset_name}__class_logits_f16.npy"
	gt_path = logits_dir / f"{args.dataset_name}__gt.npy"

	if not mask_path.exists() or not class_path.exists() or not gt_path.exists():
		raise FileNotFoundError(f"Missing cache files in: {logits_dir}")

	mask_logits = np.load(mask_path)	# [N,Q,h,w] float16
	class_logits = np.load(class_path)	# [N,Q,C(+1)] float16
	gt = np.load(gt_path)				# [N,H,W] uint8 (resized run-size)

	ood_mask = (gt == 1)
	ind_mask = (gt == 0)

	if ood_mask.sum() == 0 or ind_mask.sum() == 0:
		raise RuntimeError("GT has no OOD or no InD pixels. Check remapping.")

	mask_t = torch.from_numpy(mask_logits.astype(np.float32))
	class_t = torch.from_numpy(class_logits.astype(np.float32))

	csv_path = sweep_dir / "metrics_sweep.csv"

	for T in T_list:
		if args.method == "rba":
			anomaly = rba_from_masks(mask_t, class_t, args.num_classes, T)	# [N,h,w]
		else:
			pixel_probs = pixel_probs_from_masks(mask_t, class_t, args.num_classes, T)	# [N,C,h,w]
			anomaly = anomaly_from_pixel_probs(pixel_probs, args.method)				# [N,h,w]

		anomaly_np = anomaly.numpy()	# [N,h,w]

		ood_out = anomaly_np[ood_mask]
		ind_out = anomaly_np[ind_mask]

		val_out = np.concatenate([ind_out, ood_out])
		val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

		auprc = float(average_precision_score(val_label, val_out))
		fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

		metrics = {
			"model": "EoMT",
			"dataset": args.dataset_name,
			"method": args.method,
			"temperature": float(T),
			"mode": args.mode,
			"num_classes": int(args.num_classes),
			"auprc": auprc,
			"fpr95": fpr95,
			"auprc_pct": auprc * 100.0,
			"fpr95_pct": fpr95 * 100.0,
			"images_used": int(gt.shape[0]),
			"source": "raw_logits_cache",
			"run_dir": str(run_root),
		}

		out_json = sweep_dir / f"T{T}__{args.method}__metrics.json"
		with open(out_json, "w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)

		append_metrics_csv(csv_path, metrics)

		print(f"[T={T}] AUPRC={metrics['auprc_pct']:.4f} | FPR95={metrics['fpr95_pct']:.4f} | saved {out_json}")

	print(f"[DONE] Sweep results in: {sweep_dir}")


if __name__ == "__main__":
	main()