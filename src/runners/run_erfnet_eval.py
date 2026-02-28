import os
import glob
import json
import csv
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor

from sklearn.metrics import average_precision_score

from eval.erfnet import ERFNet
from src.utils.ood_dataset import load_ood_mask, has_ood_pixels
from src.utils.ood_metrics import fpr_at_95_tpr


NUM_CLASSES = 20


def set_torch_determinism(mode: str) -> None:
	"""
	Professor script usually does:
	cudnn.deterministic=True, cudnn.benchmark=True (contradictory but we keep it in prof-exact).
	"""
	mode = mode.lower()
	if mode == "prof-exact":
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True
	else:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def load_erfnet(weights_path: str, device: torch.device) -> torch.nn.Module:
	model = ERFNet(NUM_CLASSES)

	if device.type == "cuda":
		model = torch.nn.DataParallel(model).cuda()
	else:
		model = model.cpu()

	state = torch.load(weights_path, map_location="cpu")
	own = model.state_dict()

	# robust load: handles keys with/without "module."
	for k, v in state.items():
		k2 = k.replace("module.", "")
		if k2 in own and own[k2].shape == v.shape:
			own[k2].copy_(v)

	model.load_state_dict(own, strict=False)
	model.eval()
	return model


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
	write_header = not csv_path.exists()
	fieldnames = list(row.keys())
	with open(csv_path, "a", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		if write_header:
			writer.writeheader()
		writer.writerow(row)


def anomaly_from_logits(logits: torch.Tensor, method: str, T: float) -> torch.Tensor:
	"""
	logits: [B,C,H,W]
	return: anomaly map [B,H,W] (higher => more OOD)
	"""
	method = method.lower()

	if method == "maxlogit":
		# post-hoc: larger negative max-logit => more anomalous
		m = logits.max(dim=1).values
		return -m

	p = F.softmax(logits / T, dim=1)

	if method == "msp":
		msp = p.max(dim=1).values
		return 1.0 - msp

	if method == "maxentropy":
		ent = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
		return ent

	raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def main():
	ap = argparse.ArgumentParser()

	ap.add_argument("--input", required=True, help="Glob images, e.g. /path/images/*.*")
	ap.add_argument("--weights", required=True, help="Path .pth ERFNet")

	ap.add_argument("--method", choices=["msp", "maxlogit", "maxentropy"], default="msp")
	ap.add_argument("--temperature", type=float, default=1.0)

	ap.add_argument("--mode", choices=["robust", "prof-exact"], default="robust")
	ap.add_argument("--cpu", action="store_true")

	# fixed ERFNet eval size (as in project baseline)
	ap.add_argument("--resize", default="512x1024", help="HxW, default 512x1024")

	# artifacts
	ap.add_argument("--dataset-name", default="dataset")
	ap.add_argument("--artifacts-dir", default="artifacts")
	ap.add_argument("--save-logits", action="store_true", help="Cache logits+gt+names for temp sweep")
	ap.add_argument("--save-anomaly-maps", action="store_true", help="Save anomaly maps per image (debug)")

	args = ap.parse_args()

	set_torch_determinism(args.mode)

	# parse resize
	hw = args.resize.lower().replace(" ", "").split("x")
	if len(hw) != 2:
		raise ValueError("--resize must be like 512x1024")
	H, W = int(hw[0]), int(hw[1])
	size_hw: Tuple[int, int] = (H, W)

	input_transform = Compose([
		Resize(size_hw, Image.BILINEAR),
		ToTensor(),
	])

	device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
	model = load_erfnet(args.weights, device)

	paths = sorted(glob.glob(os.path.expanduser(args.input)))
	if len(paths) == 0:
		raise FileNotFoundError(f"Nessuna immagine trovata: {args.input}")

	art_dirs = ensure_artifact_dirs(Path(args.artifacts_dir))

	anomaly_list: List[np.ndarray] = []
	ood_list: List[np.ndarray] = []

	logits_cache: List[np.ndarray] = []
	gt_cache: List[np.ndarray] = []
	img_names: List[str] = []

	for path in paths:
		try:
			ood = load_ood_mask(path, size_hw=size_hw)
		except Exception as e:
			print(f"[SKIP] GT error {path}: {e}")
			continue

		if not has_ood_pixels(ood):
			continue

		img = Image.open(path).convert("RGB")
		x = input_transform(img).unsqueeze(0).float().to(device)	# [1,3,H,W]

		logits = model(x)  # [1,C,H,W]

		if args.save_logits:
			logits_cache.append(logits.squeeze(0).detach().cpu().numpy().astype(np.float32))
			gt_cache.append(ood.astype(np.uint8))
			img_names.append(os.path.basename(path))

		anomaly_t = anomaly_from_logits(logits, args.method, args.temperature)	# [1,H,W]
		anomaly = anomaly_t.squeeze(0).detach().cpu().float().numpy()

		ood_list.append(ood)
		anomaly_list.append(anomaly)

		if args.save_anomaly_maps:
			out_map = art_dirs["anomaly_maps"] / f"{args.dataset_name}__ERFNet__{args.method}__T{args.temperature}__{os.path.basename(path)}.npy"
			np.save(out_map, anomaly.astype(np.float32))

	if len(ood_list) == 0:
		raise RuntimeError("Nessuna immagine valida trovata (tutte skip o senza pixel OOD).")

	ood_gts = np.array(ood_list)				# [N,H,W]
	anomaly_scores = np.array(anomaly_list)		# [N,H,W]

	ood_mask = (ood_gts == 1)
	in_mask = (ood_gts == 0)

	ood_out = anomaly_scores[ood_mask]
	in_out = anomaly_scores[in_mask]

	val_out = np.concatenate([in_out, ood_out])
	val_label = np.concatenate([np.zeros(len(in_out)), np.ones(len(ood_out))])

	auprc = float(average_precision_score(val_label, val_out))
	fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

	print("=====================================")
	print(f"ERFNet | dataset={args.dataset_name} | method={args.method} | T={args.temperature} | mode={args.mode}")
	print(f"AUPRC: {auprc*100.0:.4f}")
	print(f"FPR@95TPR: {fpr95*100.0:.4f}")
	print(f"Images used: {len(ood_list)}")
	print(f"Resize: {H}x{W}")
	print("=====================================")

	metrics = {
		"model": "ERFNet",
		"dataset": args.dataset_name,
		"method": args.method,
		"temperature": float(args.temperature),
		"mode": args.mode,
		"resize_h": H,
		"resize_w": W,
		"auprc": float(auprc),
		"fpr95": float(fpr95),
		"auprc_pct": float(auprc * 100.0),
		"fpr95_pct": float(fpr95 * 100.0),
		"images_used": int(len(ood_list)),
		"input_glob": args.input,
		"weights": args.weights,
	}

	json_path = art_dirs["results"] / f"{args.dataset_name}__ERFNet__{args.method}__T{args.temperature}__{args.mode}.json"
	with open(json_path, "w") as f:
		json.dump(metrics, f, indent=2)
	print(f"[SAVED] {json_path}")

	csv_path = art_dirs["results"] / "metrics.csv"
	append_metrics_csv(csv_path, metrics)
	print(f"[UPDATED] {csv_path}")

	if args.save_logits and len(logits_cache) > 0:
		logits_path = art_dirs["logits"] / f"{args.dataset_name}__erfnet_logits_f32.npy"
		gt_path = art_dirs["logits"] / f"{args.dataset_name}__gt_u8.npy"
		names_path = art_dirs["logits"] / f"{args.dataset_name}__names.json"

		np.save(logits_path, np.stack(logits_cache, axis=0))	# [N,C,H,W]
		np.save(gt_path, np.stack(gt_cache, axis=0))			# [N,H,W]
		with open(names_path, "w") as f:
			json.dump(img_names, f, indent=2)

		print(f"[CACHED] {logits_path}")
		print(f"[CACHED] {gt_path}")
		print(f"[CACHED] {names_path}")


if __name__ == "__main__":
	main()