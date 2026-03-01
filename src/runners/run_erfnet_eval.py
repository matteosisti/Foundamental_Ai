# src/runners/run_erfnet_eval.py
#
# Unified ERFNet OOD evaluation runner (robust + prof-exact) with:
# - deterministic control via src.utils.determinism.apply_determinism(mode, seed, deterministic)
# - artifacts folder auto-run dir + config.json via src.utils.artifacts.create_run_dir
# - optional caching of RAW logits for temperature sweeps

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
from sklearn.metrics import average_precision_score

from eval.erfnet import ERFNet

from src.utils.artifacts import create_run_dir
from src.utils.ood_metrics import fpr_at_95_tpr
from src.utils.determinism import apply_determinism


NUM_CLASSES = 20


def _weights_meta(weights_path: str) -> dict:
	p = Path(os.path.expanduser(weights_path))
	meta = {
		"weights": str(p),
		"weights_basename": p.name,
	}
	try:
		st = p.stat()
		meta["weights_size_bytes"] = int(st.st_size)
		meta["weights_mtime"] = float(st.st_mtime)
	except Exception:
		pass
	return meta


def load_erfnet(weights_path: str, device: torch.device, mode: str) -> torch.nn.Module:
	model = ERFNet(NUM_CLASSES).to(device)

	state = torch.load(os.path.expanduser(weights_path), map_location="cpu")
	if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
		state = state["state_dict"]

	own = model.state_dict()

	if mode == "prof-exact":
		loadable = {}
		for k, v in state.items():
			k2 = k.replace("module.", "")
			if k2 in own and own[k2].shape == v.shape:
				loadable[k2] = v
		model.load_state_dict(loadable, strict=False)
	else:
		for k, v in state.items():
			k2 = k.replace("module.", "")
			if k2 in own and own[k2].shape == v.shape:
				own[k2].copy_(v)
		model.load_state_dict(own, strict=False)

	model.eval()
	return model


def gt_path_from_image(path_img: str) -> str:
	path_gt = path_img.replace("images", "labels_masks")
	root = path_gt

	if "RoadObstacle21" in root or "RoadObsticle21" in root:
		return os.path.splitext(root)[0] + ".png"
	if "fs_static" in root:
		return os.path.splitext(root)[0] + ".png"
	if "RoadAnomaly21" in root or "RoadAnomaly" in root:
		return os.path.splitext(root)[0] + ".png"
	if "LostAndFound" in root or "FS_LostFound_full" in root:
		return os.path.splitext(root)[0] + ".png"

	ext = os.path.splitext(root)[1].lower()
	return root if ext else (root + ".png")


def _is_already_binary_ood_mask(uvals: np.ndarray) -> bool:
	# Typical final format: 0=in, 1=ood, 255=ignore
	s = set([int(x) for x in uvals.tolist()])
	return s.issubset({0, 1, 255}) and (1 in s or 0 in s)


def remap_ood_mask(path_gt: str, ood: np.ndarray) -> np.ndarray:
	# RoadAnomaly: OOD encoded as 2 -> map to 1
	if "RoadAnomaly" in path_gt:
		ood = np.where((ood == 2), 1, ood)

	# LostAndFound:
	# Some exports already are {0,1,255}. If so: do NOTHING (keep as-is).
	# Otherwise apply legacy mapping used by older benchmark conversions.
	if "LostAndFound" in path_gt or "FS_LostFound_full" in path_gt:
		u_before = np.unique(ood)
		if not _is_already_binary_ood_mask(u_before):
			# legacy mapping
			ood = np.where((ood == 0), 255, ood)
			ood = np.where((ood == 1), 0, ood)
			ood = np.where((ood > 1) & (ood < 201), 1, ood)

	# StreetHazards mapping (if present)
	if "Streethazard" in path_gt:
		ood = np.where((ood == 14), 255, ood)
		ood = np.where((ood < 20), 0, ood)
		ood = np.where((ood == 255), 1, ood)

	return ood


def load_ood_mask(path_img: str, target_transform) -> np.ndarray:
	path_gt = gt_path_from_image(path_img)
	mask = Image.open(path_gt)
	mask = target_transform(mask)
	ood = np.array(mask)
	ood = remap_ood_mask(path_gt, ood)
	return ood


def anomaly_from_logits(logits: torch.Tensor, method: str, T: float) -> np.ndarray:
	if method == "maxlogit":
		m = logits.max(dim=1).values
		return (-m).squeeze(0).detach().cpu().float().numpy()

	p = F.softmax(logits / T, dim=1)

	if method == "msp":
		msp = p.max(dim=1).values
		return (1.0 - msp).squeeze(0).detach().cpu().float().numpy()

	if method == "maxentropy":
		ent = -(p * (p.clamp_min(1e-12)).log()).sum(dim=1)
		return ent.squeeze(0).detach().cpu().float().numpy()

	raise ValueError(f"Unknown method: {method}")


def append_metrics_csv(csv_path: Path, row: dict) -> None:
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

	ap.add_argument("--input", required=True, help="Glob images, e.g. /path/images/*.*")
	ap.add_argument("--weights", required=True, help="Path .pth ERFNet")
	ap.add_argument("--dataset-name", required=True, help="Short name e.g. RA21")
	ap.add_argument("--method", choices=["msp", "maxlogit", "maxentropy"], default="msp")
	ap.add_argument("--temperature", type=float, default=1.0)
	ap.add_argument("--mode", choices=["robust", "prof-exact"], default="robust")

	ap.add_argument("--cpu", action="store_true")

	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--deterministic", action="store_true")

	ap.add_argument("--artifacts-dir", default="artifacts")
	ap.add_argument("--save-logits", action="store_true")
	ap.add_argument("--save-anomaly-maps", action="store_true")

	args = ap.parse_args()

	if args.mode == "robust":
		want_determinism = True
	else:
		want_determinism = bool(args.deterministic)
	if args.deterministic:
		want_determinism = True

	apply_determinism(mode=args.mode, seed=int(args.seed), deterministic=bool(want_determinism))

	device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

	resize_h, resize_w = 512, 1024
	input_transform = Compose([
		Resize((resize_h, resize_w), Image.BILINEAR),
		ToTensor(),
	])
	target_transform = Compose([
		Resize((resize_h, resize_w), Image.NEAREST),
	])

	extra = {
		"input_glob": args.input,
		"resize_h": int(resize_h),
		"resize_w": int(resize_w),
		"num_classes": int(NUM_CLASSES),
		"device": str(device),
		"seed": int(args.seed),
		"deterministic": bool(want_determinism),
		"cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
		"cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
	}
	extra.update(_weights_meta(args.weights))

	art = create_run_dir(
		artifacts_root=args.artifacts_dir,
		dataset=args.dataset_name,
		model="ERFNet",
		method=args.method,
		temperature=args.temperature,
		mode=args.mode,
		extra=extra,
	)
	print("[ARTIFACTS]", art.root)

	model = load_erfnet(args.weights, device=device, mode=args.mode)

	paths = sorted(glob.glob(os.path.expanduser(args.input)))
	if len(paths) == 0:
		raise FileNotFoundError(f"No images found: {args.input}")

	anomaly_list = []
	ood_list = []

	logits_cache = []
	gt_cache = []
	names_cache = []

	# debug: track unique labels seen
	unique_before_all = set()
	unique_after_all = set()

	for path in paths:
		try:
			path_gt = gt_path_from_image(path)
			raw = np.array(target_transform(Image.open(path_gt)))
			unique_before_all.update([int(x) for x in np.unique(raw).tolist()])

			ood = load_ood_mask(path, target_transform=target_transform)
			unique_after_all.update([int(x) for x in np.unique(ood).tolist()])
		except Exception as e:
			print(f"[SKIP] GT error {path}: {e}")
			continue

		if 1 not in np.unique(ood):
			continue

		img = Image.open(path).convert("RGB")
		x = input_transform(img).unsqueeze(0).float().to(device)

		logits = model(x)

		if args.save_logits:
			logits_cache.append(logits.squeeze(0).detach().cpu().numpy().astype(np.float32))
			gt_cache.append(ood.astype(np.uint8))
			names_cache.append(os.path.basename(path))

		anomaly = anomaly_from_logits(logits, args.method, args.temperature)

		ood_list.append(ood)
		anomaly_list.append(anomaly)

		if args.save_anomaly_maps:
			out_map = art.anomaly_maps / f"{os.path.basename(path)}.npy"
			np.save(out_map, anomaly.astype(np.float32))

	if len(ood_list) == 0:
		raise RuntimeError(
			"No valid images used (all skipped or without OOD pixels). "
			f"mask_unique_before={sorted(list(unique_before_all))} "
			f"mask_unique_after={sorted(list(unique_after_all))}"
		)

	ood_gts = np.array(ood_list)
	anomaly_scores = np.array(anomaly_list)

	ood_mask = (ood_gts == 1)
	ind_mask = (ood_gts == 0)

	ood_out = anomaly_scores[ood_mask]
	ind_out = anomaly_scores[ind_mask]

	val_out = np.concatenate([ind_out, ood_out])
	val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

	auprc = float(average_precision_score(val_label, val_out))
	fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

	metrics = {
		"model": "ERFNet",
		"dataset": args.dataset_name,
		"method": args.method,
		"temperature": float(args.temperature),
		"mode": args.mode,
		"seed": int(args.seed),
		"deterministic": bool(want_determinism),
		"auprc": auprc,
		"fpr95": fpr95,
		"auprc_pct": auprc * 100.0,
		"fpr95_pct": fpr95 * 100.0,
		"images_used": int(len(ood_list)),
		"input_glob": args.input,
		"weights": os.path.expanduser(args.weights),
		"device": str(device),
		"resize_h": int(resize_h),
		"resize_w": int(resize_w),
		"gt_h": int(ood_gts.shape[-2]),
		"gt_w": int(ood_gts.shape[-1]),
		"num_classes": int(NUM_CLASSES),
		"cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
		"cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
		"mask_unique_before": sorted(list(unique_before_all)),
		"mask_unique_after": sorted(list(unique_after_all)),
	}

	print("=====================================")
	print(f"ERFNet | dataset={args.dataset_name} | method={args.method} | T={args.temperature} | mode={args.mode}")
	print(f"AUPRC: {metrics['auprc_pct']:.4f}")
	print(f"FPR@95TPR: {metrics['fpr95_pct']:.4f}")
	print(f"Images used: {metrics['images_used']}")
	print("=====================================")

	json_path = art.results / "metrics.json"
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	print(f"[SAVED] {json_path}")

	csv_path = art.results / "metrics.csv"
	append_metrics_csv(csv_path, metrics)
	print(f"[SAVED] {csv_path}")

	if args.save_logits and len(logits_cache) > 0:
		np.save(art.logits / f"{args.dataset_name}__logits.npy", np.stack(logits_cache, axis=0))
		np.save(art.logits / f"{args.dataset_name}__gt.npy", np.stack(gt_cache, axis=0))
		with open(art.logits / f"{args.dataset_name}__names.json", "w", encoding="utf-8") as f:
			json.dump(names_cache, f, indent=2)

		print(f"[CACHED] {art.logits / f'{args.dataset_name}__logits.npy'}")
		print(f"[CACHED] {art.logits / f'{args.dataset_name}__gt.npy'}")
		print(f"[CACHED] {art.logits / f'{args.dataset_name}__names.json'}")


if __name__ == "__main__":
	main()