# src/runners/sweep_temp_from_cache.py
#
# Temperature sweep for ERFNet using cached logits.
#
# It can:
# - use an explicit --run-dir
# - OR auto-pick the latest run matching (method, mode) via run-dir name pattern:
#     2026-03-01_12-28-22__msp__T1.0__robust__559e7ee5
#
# Expected cache files produced by run_erfnet_eval.py:
#   logits/<DATASET>__logits.npy   [N,C,H,W]
#   logits/<DATASET>__gt.npy       [N,H,W]
#
# Example:
#   python3 -m src.runners.sweep_temp_from_cache \
#     --dataset-name RA21 \
#     --artifacts-dir "/content/drive/MyDrive/anom_project/artifacts" \
#     --use-latest \
#     --method msp \
#     --mode robust

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


# New pretty run-dir naming:
#   <ts>__<method>__T<T>__<mode>__<hash>
RUN_RE = re.compile(
	r"^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
	r"__"
	r"(?P<method>[a-zA-Z0-9\-]+)"
	r"__T(?P<T>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
	r"__"
	r"(?P<mode>robust|prof-exact)"
	r"__"
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
	return {
		"ts": d["ts"],  # lexicographically sortable
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
			l_path = logits_dir / f"{dataset}__logits.npy"
			g_path = logits_dir / f"{dataset}__gt.npy"
			if not l_path.exists() or not g_path.exists():
				continue

		candidates.append((info["ts"], run_dir))

	if not candidates:
		raise FileNotFoundError(
			f"No matching runs found in {base} for method={method}, mode={mode} (require_logits={require_logits})."
		)

	# latest timestamp
	candidates.sort(key=lambda x: x[0], reverse=True)
	return candidates[0][1]


def main():
	ap = argparse.ArgumentParser(description="Temperature sweep from cached ERFNet logits.")
	ap.add_argument("--dataset-name", required=True, help="e.g. RA21")
	ap.add_argument("--artifacts-dir", default="artifacts", help="Root artifacts folder")

	ap.add_argument("--mode", choices=["robust", "prof-exact"], required=True)
	ap.add_argument("--method", choices=["msp", "maxentropy", "maxlogit"], required=True)

	ap.add_argument("--temperatures", default="0.5,0.75,1.0,1.1,1.25,1.5,2.0", help="comma-separated")

	ap.add_argument("--use-latest", action="store_true", help="Auto-pick latest run matching method+mode")
	ap.add_argument("--run-dir", default=None, help="Explicit run directory path")

	args = ap.parse_args()

	T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

	# Resolve run directory
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

	# Load cache
	l_path = logits_dir / f"{args.dataset_name}__logits.npy"
	g_path = logits_dir / f"{args.dataset_name}__gt.npy"

	if not l_path.exists() or not g_path.exists():
		raise FileNotFoundError(f"Missing cache files: {l_path} or {g_path}")

	logits = np.load(l_path)  # [N,C,H,W] float32
	gt = np.load(g_path)      # [N,H,W] uint8

	ood_mask = (gt == 1)
	ind_mask = (gt == 0)

	if ood_mask.sum() == 0 or ind_mask.sum() == 0:
		raise RuntimeError("GT has no OOD or no InD pixels. Check remapping.")

	# torch once (CPU)
	logits_t = torch.from_numpy(logits)

	csv_path = sweep_dir / "metrics_sweep.csv"

	for T in T_list:
		if args.method == "msp":
			p = F.softmax(logits_t / T, dim=1)                    # [N,C,H,W]
			m = torch.max(p, dim=1).values                        # [N,H,W]
			anomaly = (1.0 - m).numpy()                           # [N,H,W]

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

		res = {
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
			"gt_h": int(gt.shape[-2]),
			"gt_w": int(gt.shape[-1]),
			"source": "logits_cache",
			"run_dir": str(run_root),
		}

		out_json = sweep_dir / f"T{T}__metrics.json"
		with open(out_json, "w", encoding="utf-8") as f:
			json.dump(res, f, indent=2)

		append_metrics_csv(csv_path, res)

		print(f"[T={T}] AUPRC={res['auprc_pct']:.4f} | FPR95={res['fpr95_pct']:.4f} | saved {out_json}")

	print(f"[DONE] Run used: {run_root}")
	print(f"[DONE] Sweep results in: {sweep_dir}")


if __name__ == "__main__":
	main()