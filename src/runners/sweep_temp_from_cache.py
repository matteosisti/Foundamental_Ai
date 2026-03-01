# src/runners/sweep_temp_from_cache.py
#
# Temperature sweep for ERFNet using cached logits.
# Picks the "latest" run directory by parsing the run-dir name:
#   YYYY-MM-DD_HH-MM-SS__<method>__T<temp>__<mode>__<hash>
#
# Usage examples:
#   python3 -m src.runners.sweep_temp_from_cache \
#     --dataset-name RA21 \
#     --artifacts-dir "/content/drive/MyDrive/anom_project/artifacts" \
#     --use-latest \
#     --method msp \
#     --mode robust
#
#   python3 -m src.runners.sweep_temp_from_cache \
#     --dataset-name RA21 \
#     --artifacts-dir "/content/drive/MyDrive/anom_project/artifacts" \
#     --use-latest \
#     --method msp \
#     --mode prof-exact
#
# Or explicit:
#   python3 -m src.runners.sweep_temp_from_cache --run-dir ".../RA21/ERFNet/<run>"

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


RUN_RE = re.compile(
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


def parse_run_dir_name(name: str) -> Optional[dict]:
	m = RUN_RE.match(name)
	if not m:
		return None
	d = m.groupdict()
	# Keep raw timestamp string; lexicographic works with this format
	return {
		"ts": d["ts"],
		"method": d["method"].lower(),
		"T": float(d["T"]),
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

		if info["method"] != method:
			continue
		if info["mode"] != mode:
			continue

		if require_logits:
			logits_dir = run_dir / "logits"
			if not logits_dir.exists():
				continue
			# We require the two core cache files
			logits_path = logits_dir / f"{dataset}__logits.npy"
			gt_path = logits_dir / f"{dataset}__gt.npy"
			if not logits_path.exists() or not gt_path.exists():
				continue

		candidates.append((info["ts"], run_dir))

	if len(candidates) == 0:
		raise FileNotFoundError(
			f"No matching runs found in {base} for method={method}, mode={mode} "
			f"(require_logits={require_logits})."
		)

	# Latest by timestamp string (format is sortable)
	candidates.sort(key=lambda x: x[0], reverse=True)
	return candidates[0][1]


def main():
	ap = argparse.ArgumentParser()

	ap.add_argument("--dataset-name", required=True, help="e.g. RA21")
	ap.add_argument("--artifacts-dir", default="artifacts", help="Root artifacts folder")

	# Must match the run-dir name tokens
	ap.add_argument("--mode", choices=["robust", "prof-exact"], required=True)
	ap.add_argument("--method", choices=["msp", "maxentropy", "maxlogit"], required=True)

	ap.add_argument("--temperatures", default="0.5,0.75,1.0,1.1,1.25,1.5,2.0", help="comma-separated")

	# selection
	ap.add_argument("--use-latest", action="store_true", help="Auto-pick latest run matching method+mode")
	ap.add_argument("--run-dir", default=None, help="Optional: manually point to a run directory")

	args = ap.parse_args()

	T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

	# pick run directory
	if args.run_dir is not None:
		run_root = Path(os.path.expanduser(args.run_dir))
	else:
		if not args.use_latest:
			raise ValueError("Provide --run-dir or use --use-latest")
		run_root = find_latest_run_dir(
			artifacts_root=args.artifacts_dir,
			dataset=args.dataset_name,
			model="ERFNet",
			method=args.method,
			mode=args.mode,
			require_logits=True,
		)

	logits_dir = run_root / "logits"
	sweep_dir = run_root / "sweep" / f"{args.method}__{args.mode}"
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

	# logits -> torch once
	logits_t = torch.from_numpy(logits)  # float32 CPU

	csv_path = sweep_dir / "metrics_sweep.csv"

	# NOTE:
	# - Only MSP temperature scaling is mathematically standard on raw logits.
	# - For maxentropy/maxlogit you can still sweep (definition-dependent), but here:
	#   * maxlogit: use -max(logits) (temperature would scale logits; ok but trivial monotonic)
	#   * maxentropy: entropy of softmax(logits/T)
	for T in T_list:
		if args.method == "msp":
			p = F.softmax(logits_t / T, dim=1)                   # [N,C,H,W]
			msp = torch.max(p, dim=1).values                     # [N,H,W]
			anomaly = (1.0 - msp).numpy()                        # [N,H,W]

		elif args.method == "maxentropy":
			p = F.softmax(logits_t / T, dim=1)
			ent = -(p * torch.clamp_min(p, 1e-12).log()).sum(dim=1)
			anomaly = ent.numpy()

		elif args.method == "maxlogit":
			m = torch.max(logits_t / T, dim=1).values
			anomaly = (-m).numpy()

		else:
			raise ValueError(f"Unknown method: {args.method}")

		ood_out = anomaly[ood_mask]
		ind_out = anomaly[ind_mask]

		val_out = np.concatenate([ind_out, ood_out])
		val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

		auprc = float(average_precision_score(val_label, val_out))
		fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

		metrics = {
			"model": "ERFNet",
			"dataset": args.dataset_name,
			"method": args.method,
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

	print(f"[DONE] Run used: {run_root}")
	print(f"[DONE] Sweep results in: {sweep_dir}")


if __name__ == "__main__":
	main()