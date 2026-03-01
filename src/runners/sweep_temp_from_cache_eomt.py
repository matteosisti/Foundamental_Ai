# src/runners/sweep_temp_from_cache_eomt.py
import os
import re
import json
import csv
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.utils.ood_metrics import fpr_at_95_tpr


RUN_RE = re.compile(
	# 2026-03-01_13-22-52__msp__T1.0__prof-exact__5e899996
	r"^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})__"
	r"(?P<method>[^_]+)__"
	r"T(?P<T>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)__"
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


def list_candidate_runs(model_root: Path) -> List[Path]:
	if not model_root.exists():
		return []
	return [p for p in model_root.iterdir() if p.is_dir()]


def parse_run_dir_name(name: str) -> Optional[dict]:
	m = RUN_RE.match(name)
	if not m:
		return None
	d = m.groupdict()
	return {
		"ts": d["ts"],
		"method": d["method"].lower(),
		"T": float(d["T"]),
		"mode": d["mode"].lower(),
		"hash": d["hash"].lower(),
	}


def resolve_latest_run_dir_filtered(
	artifacts_root: str,
	dataset: str,
	model: str,
	method: str,
	mode: str,
) -> Path:
	root = Path(os.path.expanduser(artifacts_root))
	model_root = root / dataset / model
	cands = list_candidate_runs(model_root)

	method = method.lower().strip()
	mode = mode.lower().strip()

	best: Optional[Path] = None
	best_ts: Optional[str] = None

	for p in cands:
		info = parse_run_dir_name(p.name)
		if info is None:
			continue
		if info["method"] != method:
			continue
		if info["mode"] != mode:
			continue

		ts = info["ts"]
		if best is None or ts > best_ts:
			best = p
			best_ts = ts

	if best is None:
		raise FileNotFoundError(
			f"No run found for dataset={dataset}, model={model}, method={method}, mode={mode} under: {model_root}"
		)
	return best


# -----------------------------
# EoMT math (same as runner)
# -----------------------------
def pixel_probs_from_masks(
	mask_logits: torch.Tensor,   # [N,Q,h,w]
	class_logits: torch.Tensor,  # [N,Q,C(+1)]
	num_classes: int,
	temperature: float,
) -> torch.Tensor:
	if class_logits.shape[-1] == num_classes + 1:
		class_logits = class_logits[..., :num_classes]

	class_prob = F.softmax(class_logits / temperature, dim=-1)  # [N,Q,C]
	mask_prob = torch.sigmoid(mask_logits)                       # [N,Q,h,w]

	pixel = torch.einsum("nqc,nqhw->nchw", class_prob, mask_prob)  # [N,C,h,w]
	den = pixel.sum(dim=1, keepdim=True).clamp_min(1e-8)
	return pixel / den


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


def rba_from_masks(
	mask_logits: torch.Tensor,   # [N,Q,h,w]
	class_logits: torch.Tensor,  # [N,Q,C(+1)]
	num_classes: int,
	temperature: float,
	area_pow: float = 0.5,
) -> torch.Tensor:
	if class_logits.shape[-1] == num_classes + 1:
		class_logits = class_logits[..., :num_classes]

	class_prob = F.softmax(class_logits / temperature, dim=-1)  # [N,Q,C]
	conf = class_prob.max(dim=-1).values                         # [N,Q]

	mask_prob = torch.sigmoid(mask_logits)                       # [N,Q,h,w]
	area = mask_prob.mean(dim=(-2, -1))                          # [N,Q]

	reliability = conf * (area.clamp_min(1e-6) ** area_pow)       # [N,Q]
	reliability = reliability.unsqueeze(-1).unsqueeze(-1)         # [N,Q,1,1]

	normality = (reliability * mask_prob).amax(dim=1)             # [N,h,w]
	return 1.0 - normality


def upsample_to_gt(anomaly: torch.Tensor, gt_hw: Tuple[int, int]) -> torch.Tensor:
	# anomaly: [N,h,w] -> [N,H,W]
	H, W = gt_hw
	if anomaly.shape[-2:] == (H, W):
		return anomaly
	return F.interpolate(
		anomaly.unsqueeze(1),  # [N,1,h,w]
		size=(H, W),
		mode="bilinear",
		align_corners=False,
	).squeeze(1)


# -----------------------------
# Main
# -----------------------------
def main():
	ap = argparse.ArgumentParser()

	ap.add_argument("--dataset-name", required=True, help="e.g. RA21")
	ap.add_argument("--artifacts-dir", default="artifacts")

	ap.add_argument("--mode", choices=["robust", "prof-exact"], required=True)
	ap.add_argument("--method", choices=["msp", "maxentropy", "maxlogit", "rba"], required=True)

	ap.add_argument("--num-classes", type=int, default=19)
	ap.add_argument("--temperatures", default="0.5,0.75,1.0,1.1,1.25,1.5,2.0")

	ap.add_argument("--use-latest", action="store_true", help="Use latest run matching method+mode")
	ap.add_argument("--run-dir", default=None, help="Manual run dir override")

	args = ap.parse_args()

	T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

	# Resolve run dir
	if args.run_dir is not None:
		run_root = Path(os.path.expanduser(args.run_dir))
	else:
		if not args.use_latest:
			raise ValueError("Provide --run-dir or use --use-latest")
		run_root = resolve_latest_run_dir_filtered(
			artifacts_root=args.artifacts_dir,
			dataset=args.dataset_name,
			model="EoMT",
			method=args.method,
			mode=args.mode,
		)

	logits_dir = run_root / "logits"
	if not logits_dir.exists():
		raise FileNotFoundError(f"Missing logits dir: {logits_dir}")

	# Sweep output folder avoids mixing
	sweep_dir = run_root / "sweep" / f"{args.method}__{args.mode}"
	sweep_dir.mkdir(parents=True, exist_ok=True)
	csv_path = sweep_dir / "metrics_sweep.csv"

	# Cache files (match run_eomt_eval.py)
	mask_path = logits_dir / f"{args.dataset_name}__mask_logits_f16.npy"
	class_path = logits_dir / f"{args.dataset_name}__class_logits_f16.npy"
	gt_path = logits_dir / f"{args.dataset_name}__gt.npy"

	if not mask_path.exists():
		raise FileNotFoundError(f"Missing: {mask_path}")
	if not class_path.exists():
		raise FileNotFoundError(f"Missing: {class_path}")
	if not gt_path.exists():
		raise FileNotFoundError(f"Missing: {gt_path}")

	mask_logits = np.load(mask_path)   # [N,Q,h,w] float16
	class_logits = np.load(class_path) # [N,Q,C(+1)] float16
	gt = np.load(gt_path)              # [N,H,W] uint8

	N = int(gt.shape[0])
	gt_H = int(gt.shape[1])
	gt_W = int(gt.shape[2])

	ood_mask = (gt == 1)
	ind_mask = (gt == 0)
	if ood_mask.sum() == 0 or ind_mask.sum() == 0:
		raise RuntimeError("GT has no OOD or no InD pixels. Check remapping / dataset.")

	# Torch once (CPU)
	mask_t = torch.from_numpy(mask_logits.astype(np.float32))    # [N,Q,h,w]
	class_t = torch.from_numpy(class_logits.astype(np.float32))  # [N,Q,C(+1)]

	logits_h = int(mask_t.shape[-2])
	logits_w = int(mask_t.shape[-1])

	for T in T_list:
		# Compute anomaly at logits resolution
		if args.method == "rba":
			anomaly = rba_from_masks(mask_t, class_t, args.num_classes, T)  # [N,h,w]
		else:
			pixel_probs = pixel_probs_from_masks(mask_t, class_t, args.num_classes, T)  # [N,C,h,w]
			anomaly = anomaly_from_pixel_probs(pixel_probs, args.method)                # [N,h,w]

		# Upsample to GT resolution (fix IndexError)
		anomaly = upsample_to_gt(anomaly, (gt_H, gt_W))  # [N,H,W]
		anomaly_np = anomaly.numpy()

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
			"images_used": int(N),
			"gt_h": int(gt_H),
			"gt_w": int(gt_W),
			"logits_h": int(logits_h),
			"logits_w": int(logits_w),
			"source": "raw_logits_cache",
			"run_dir": str(run_root),
		}

		out_json = sweep_dir / f"T{T}__metrics.json"
		with open(out_json, "w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)

		append_metrics_csv(csv_path, metrics)

		print(f"[T={T}] AUPRC={metrics['auprc_pct']:.4f} | FPR95={metrics['fpr95_pct']:.4f} | saved {out_json}")

	print(f"[DONE] Run used: {run_root}")
	print(f"[DONE] Sweep results in: {sweep_dir}")


if __name__ == "__main__":
	main()