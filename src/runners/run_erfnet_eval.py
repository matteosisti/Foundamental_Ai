# =========================
# File: scripts/erfnet_eval.py
# Full rewrite + --prof-exact mode
# =========================
import os
import glob
import json
import csv
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor

from eval.erfnet import ERFNet
from sklearn.metrics import average_precision_score

# Robust import: works whether you run from repo root or as package
try:
	from src.utils.ood_metrics import fpr_at_95_tpr
except Exception:
	from ood_metrics import fpr_at_95_tpr  # fallback

NUM_CLASSES = 20

# Default resize chosen by your code (keep it stable)
input_transform = Compose([
	Resize((512, 1024), Image.BILINEAR),
	ToTensor(),
])

target_transform = Compose([
	Resize((512, 1024), Image.NEAREST),
])


def load_erfnet(weights_path: str, device: torch.device) -> torch.nn.Module:
	"""
	Load ERFNet with a robust state_dict merge (handles module. prefixes).
	Prints load stats to diagnose partial loads (common mismatch source).
	"""
	model = ERFNet(NUM_CLASSES)

	if device.type == "cuda":
		model = torch.nn.DataParallel(model).cuda()
	else:
		model = model.cpu()

	state = torch.load(weights_path, map_location="cpu")
	# If checkpoint is nested (rare), unwrap
	if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
		state = state["state_dict"]

	own = model.state_dict()

	loadable = 0
	for k, v in state.items():
		k2 = k.replace("module.", "")
		if k2 in own and hasattr(v, "shape") and own[k2].shape == v.shape:
			own[k2].copy_(v)
			loadable += 1

	model.load_state_dict(own, strict=False)
	model.eval()

	print(f"[ERFNet] Loaded weights: {weights_path}")
	print(f"[ERFNet] loadable keys: {loadable} / model keys: {len(own)}")
	return model


def gt_path_from_image(path_img: str) -> str:
	path_gt = path_img.replace("images", "labels_masks")
	root = path_gt
	ext = os.path.splitext(root)[1].lower()

	# Many GTs are .png even when images are .jpg/.webp
	if "RoadObstacle21" in root or "RoadObsticle21" in root:
		return os.path.splitext(root)[0] + ".png"
	if "fs_static" in root:
		return os.path.splitext(root)[0] + ".png"
	if "RoadAnomaly21" in root or "RoadAnomaly" in root:
		return os.path.splitext(root)[0] + ".png"
	if "LostAndFound" in root or "FS_LostFound_full" in root:
		return os.path.splitext(root)[0] + ".png"

	return path_gt if ext else (root + ".png")


def load_ood_mask(path_img: str) -> np.ndarray:
	path_gt = gt_path_from_image(path_img)
	mask = Image.open(path_gt)
	mask = target_transform(mask)
	ood = np.array(mask)

	# remapping as in original scripts
	if "RoadAnomaly" in path_gt:
		ood = np.where((ood == 2), 1, ood)

	if "LostAndFound" in path_gt or "FS_LostFound_full" in path_gt:
		ood = np.where((ood == 0), 255, ood)
		ood = np.where((ood == 1), 0, ood)
		ood = np.where((ood > 1) & (ood < 201), 1, ood)

	if "Streethazard" in path_gt:
		ood = np.where((ood == 14), 255, ood)
		ood = np.where((ood < 20), 0, ood)
		ood = np.where((ood == 255), 1, ood)

	return ood


def anomaly_from_logits(logits: torch.Tensor, method: str, T: float) -> np.ndarray:
	"""
	logits: [1, C, H, W]
	return: anomaly map [H, W] (numpy)
	"""
	if method == "maxlogit":
		m = logits.max(dim=1).values  # [1,H,W]
		return (-m).squeeze(0).detach().cpu().float().numpy()

	p = F.softmax(logits / T, dim=1)

	if method == "msp":
		msp = p.max(dim=1).values
		return (1.0 - msp).squeeze(0).detach().cpu().float().numpy()

	if method == "maxentropy":
		ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=1)
		return ent.squeeze(0).detach().cpu().float().numpy()

	raise ValueError(f"Unknown method: {method}")


def anomaly_prof_exact(result: torch.Tensor) -> np.ndarray:
	"""
	Professor baseline (as seen in evalAnomaly.py-like scripts):
	anomaly = 1 - max(result, axis=0), WITHOUT softmax.
	result is assumed to be [1, C, H, W] or [C, H, W] compatible.

	Returns [H,W] float32 numpy.
	"""
	if result.dim() == 4:
		r = result.squeeze(0)
	elif result.dim() == 3:
		r = result
	else:
		raise ValueError(f"Unexpected ERFNet output shape: {tuple(result.shape)}")

	# raw max over channels
	m = torch.max(r, dim=0).values  # [H,W]
	anomaly = 1.0 - m
	return anomaly.detach().cpu().float().numpy()


def ensure_artifact_dirs(root: Path) -> dict:
	dirs = {
		"root": root,
		"results": root / "results",
		"logits": root / "logits",
		"anomaly_maps": root / "anomaly_maps",
	}
	for p in dirs.values():
		if isinstance(p, Path):
			p.mkdir(parents=True, exist_ok=True)
	return dirs


def append_metrics_csv(csv_path: Path, row: dict) -> None:
	"""
	Appends a single row to metrics.csv (creates file with header if missing).
	"""
	write_header = not csv_path.exists()
	fieldnames = list(row.keys())

	with open(csv_path, "a", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		if write_header:
			writer.writeheader()
		writer.writerow(row)


@torch.no_grad()
def main():
	ap = argparse.ArgumentParser()

	ap.add_argument("--input", required=True, help="Glob, es: /path/images/*.*")
	ap.add_argument("--weights", required=True, help="Path .pth ERFNet")
	ap.add_argument("--cpu", action="store_true")

	# Our methods (softmax-based)
	ap.add_argument("--method", choices=["msp", "maxlogit", "maxentropy"], default="msp")
	ap.add_argument("--temperature", type=float, default=1.0)

	# Professor-identical mode
	ap.add_argument(
		"--prof-exact",
		action="store_true",
		help="Replica baseline professor: anomaly = 1 - max(raw_output) (no softmax). "
			 "Also offers --prof-permute-bug to mimic broken tensor permute if needed."
	)
	ap.add_argument(
		"--prof-permute-bug",
		action="store_true",
		help="Bug-compatible: apply x = x.permute(0,3,1,2) AFTER ToTensor(). "
			 "Use only if you must match professor numbers exactly."
	)

	# Saving / artifacts
	ap.add_argument("--dataset-name", default="dataset", help="Name used in saved artifacts filenames")
	ap.add_argument("--artifacts-dir", default="artifacts", help="Root artifacts folder (results/logits/...)")
	ap.add_argument("--save-logits", action="store_true", help="Save logits+gt for temperature scaling reuse")
	ap.add_argument("--save-anomaly-maps", action="store_true", help="Save anomaly maps per image (debug)")

	args = ap.parse_args()

	device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
	model = load_erfnet(args.weights, device)

	paths = sorted(glob.glob(os.path.expanduser(args.input)))
	if len(paths) == 0:
		raise FileNotFoundError(f"Nessuna immagine trovata: {args.input}")

	art_dirs = ensure_artifact_dirs(Path(args.artifacts_dir))

	# Collect for metrics
	anomaly_list = []
	ood_list = []

	# Optional caches
	logits_cache = []   # list of [C,H,W] float32
	gt_cache = []       # list of [H,W] uint8
	img_names = []      # to keep alignment

	for path in paths:
		try:
			ood = load_ood_mask(path)
		except Exception as e:
			print(f"[SKIP] GT error {path}: {e}")
			continue

		# If no anomaly pixel, skip (same logic as original)
		if 1 not in np.unique(ood):
			continue

		img = Image.open(path).convert("RGB")
		x = input_transform(img).unsqueeze(0).float().to(device)  # [1,3,H,W]

		# Bug-compatible hook (ONLY if needed)
		if args.prof_exact and args.prof_permute_bug:
			# This is intentionally wrong for ToTensor output, but may match professor baseline behavior.
			x = x.permute(0, 3, 1, 2)

		out = model(x)  # expected [1,C,H,W] in our pipeline

		# Save logits/gt for temperature scaling (only once, then reuse)
		# For prof_exact, we still cache raw outputs (same tensor) for later analysis.
		if args.save_logits:
			if out.dim() == 4:
				logits_cache.append(out.squeeze(0).detach().cpu().numpy().astype(np.float32))
			elif out.dim() == 3:
				logits_cache.append(out.detach().cpu().numpy().astype(np.float32))
			else:
				raise ValueError(f"Unexpected ERFNet output shape for caching: {tuple(out.shape)}")
			gt_cache.append(ood.astype(np.uint8))
			img_names.append(os.path.basename(path))

		# Compute anomaly
		if args.prof_exact:
			anomaly = anomaly_prof_exact(out)
			method_name = "prof_exact"
			T_used = None
		else:
			anomaly = anomaly_from_logits(out, args.method, args.temperature)
			method_name = args.method
			T_used = args.temperature

		ood_list.append(ood)
		anomaly_list.append(anomaly)

		if args.save_anomaly_maps:
			T_tag = "NA" if T_used is None else f"{T_used}"
			out_map = art_dirs["anomaly_maps"] / f"{args.dataset_name}__{method_name}__T{T_tag}__{os.path.basename(path)}.npy"
			np.save(out_map, anomaly.astype(np.float32))

	# Safety
	if len(ood_list) == 0:
		raise RuntimeError("No valid images found (all skipped or without OOD pixels).")

	ood_gts = np.array(ood_list)
	anomaly_scores = np.array(anomaly_list)

	ood_mask = (ood_gts == 1)
	in_mask = (ood_gts == 0)

	ood_out = anomaly_scores[ood_mask]
	in_out = anomaly_scores[in_mask]

	val_out = np.concatenate([in_out, ood_out])
	val_label = np.concatenate([np.zeros(len(in_out)), np.ones(len(ood_out))])

	auprc = average_precision_score(val_label, val_out)
	fpr95 = fpr_at_95_tpr(val_out, val_label)

	# Print
	print("=====================================")
	if args.prof_exact:
		print(f"ERFNet | dataset={args.dataset_name} | method=prof_exact | T=NA | permute_bug={args.prof_permute_bug}")
	else:
		print(f"ERFNet | dataset={args.dataset_name} | method={args.method} | T={args.temperature}")
	print(f"AUPRC: {auprc*100.0:.4f}")
	print(f"FPR@95TPR: {fpr95*100.0:.4f}")
	print(f"Images used: {len(ood_list)}")
	print("=====================================")

	# Save metrics JSON
	metrics = {
		"model": "ERFNet",
		"dataset": args.dataset_name,
		"method": ("prof_exact" if args.prof_exact else args.method),
		"temperature": (None if args.prof_exact else float(args.temperature)),
		"auprc": float(auprc),
		"fpr95": float(fpr95),
		"auprc_pct": float(auprc * 100.0),
		"fpr95_pct": float(fpr95 * 100.0),
		"images_used": int(len(ood_list)),
		"input_glob": args.input,
		"weights": args.weights,
		"prof_permute_bug": bool(args.prof_permute_bug) if args.prof_exact else False,
	}

	T_tag = "NA" if args.prof_exact else str(args.temperature)
	method_tag = "prof_exact" if args.prof_exact else args.method

	json_path = art_dirs["results"] / f"{args.dataset_name}__ERFNet__{method_tag}__T{T_tag}.json"
	with open(json_path, "w") as f:
		json.dump(metrics, f, indent=2)
	print(f"[SAVED] {json_path}")

	# Append CSV (single global file)
	csv_path = art_dirs["results"] / "metrics.csv"
	append_metrics_csv(csv_path, metrics)
	print(f"[UPDATED] {csv_path}")

	# Save logits cache (optional)
	if args.save_logits and len(logits_cache) > 0:
		logits_path = art_dirs["logits"] / f"{args.dataset_name}__logits.npy"
		gt_path = art_dirs["logits"] / f"{args.dataset_name}__gt.npy"
		names_path = art_dirs["logits"] / f"{args.dataset_name}__names.json"

		np.save(logits_path, np.stack(logits_cache, axis=0))  # [N,C,H,W]
		np.save(gt_path, np.stack(gt_cache, axis=0))          # [N,H,W]
		with open(names_path, "w") as f:
			json.dump(img_names, f, indent=2)

		print(f"[CACHED] {logits_path}")
		print(f"[CACHED] {gt_path}")
		print(f"[CACHED] {names_path}")


if __name__ == "__main__":
	main()