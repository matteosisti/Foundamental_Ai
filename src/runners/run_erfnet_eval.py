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


NUM_CLASSES = 20


def set_determinism(mode: str) -> None:
	# prof-exact: prof code often has active benchmark (not deterministic but more ‘similar’)
	if mode == "prof-exact":
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.deterministic = False
	else:
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True


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

	# fallback
	ext = os.path.splitext(root)[1].lower()
	return root if ext else (root + ".png")


def remap_ood_mask(path_gt: str, ood: np.ndarray) -> np.ndarray:
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


def load_ood_mask(path_img: str, target_transform) -> np.ndarray:
	path_gt = gt_path_from_image(path_img)
	mask = Image.open(path_gt)
	mask = target_transform(mask)
	ood = np.array(mask)
	ood = remap_ood_mask(path_gt, ood)
	return ood


def anomaly_from_logits(logits: torch.Tensor, method: str, T: float) -> np.ndarray:
	# logits: [1,C,H,W] -> anomaly [H,W]
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

	# Artifacts
	ap.add_argument("--artifacts-dir", default="artifacts", help="Root artifacts folder (can be Drive)")
	ap.add_argument("--save-logits", action="store_true", help="Cache logits+gt for temperature sweep")
	ap.add_argument("--save-anomaly-maps", action="store_true", help="Save anomaly maps per image (debug)")

	args = ap.parse_args()

	set_determinism(args.mode)

	device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

	# fixed transforms (as you already used)
	input_transform = Compose([
		Resize((512, 1024), Image.BILINEAR),
		ToTensor(),
	])
	target_transform = Compose([
		Resize((512, 1024), Image.NEAREST),
	])

	# create run dir automatically
	art = create_run_dir(
		artifacts_root=args.artifacts_dir,
		dataset=args.dataset_name,
		model="ERFNet",
		method=args.method,
		temperature=args.temperature,
		mode=args.mode,
		extra={
			"weights": args.weights,
			"input_glob": args.input,
			"resize_h": 512,
			"resize_w": 1024,
			"num_classes": NUM_CLASSES,
		},
	)

	print("[ARTIFACTS]", art.root)

	model = load_erfnet(args.weights, device)

	paths = sorted(glob.glob(os.path.expanduser(args.input)))
	if len(paths) == 0:
		raise FileNotFoundError(f"No images found: {args.input}")

	anomaly_list = []
	ood_list = []

	logits_cache = []
	gt_cache = []
	names_cache = []

	for path in paths:
		try:
			ood = load_ood_mask(path, target_transform=target_transform)
		except Exception as e:
			print(f"[SKIP] GT error {path}: {e}")
			continue

		# skip if no OOD pixel (same as professor scripts typically)
		if 1 not in np.unique(ood):
			continue

		img = Image.open(path).convert("RGB")
		x = input_transform(img).unsqueeze(0).float().to(device)

		logits = model(x)  # [1,C,H,W]

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
		raise RuntimeError("No valid images used (all skipped or without OOD pixels).")

	ood_gts = np.array(ood_list)            # [N,H,W]
	anomaly_scores = np.array(anomaly_list) # [N,H,W]

	ood_mask = (ood_gts == 1)
	in_mask = (ood_gts == 0)

	ood_out = anomaly_scores[ood_mask]
	in_out = anomaly_scores[in_mask]

	val_out = np.concatenate([in_out, ood_out])
	val_label = np.concatenate([np.zeros(len(in_out)), np.ones(len(ood_out))])

	auprc = float(average_precision_score(val_label, val_out))
	fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

	metrics = {
		"model": "ERFNet",
		"dataset": args.dataset_name,
		"method": args.method,
		"temperature": float(args.temperature),
		"mode": args.mode,
		"auprc": auprc,
		"fpr95": fpr95,
		"auprc_pct": auprc * 100.0,
		"fpr95_pct": fpr95 * 100.0,
		"images_used": int(len(ood_list)),
		"input_glob": args.input,
		"weights": args.weights,
		"device": str(device),
	}

	print("=====================================")
	print(f"ERFNet | dataset={args.dataset_name} | method={args.method} | T={args.temperature} | mode={args.mode}")
	print(f"AUPRC: {metrics['auprc_pct']:.4f}")
	print(f"FPR@95TPR: {metrics['fpr95_pct']:.4f}")
	print(f"Images used: {metrics['images_used']}")
	print("=====================================")

	# save metrics
	json_path = art.results / "metrics.json"
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	print(f"[SAVED] {json_path}")

	csv_path = art.results / "metrics.csv"
	append_metrics_csv(csv_path, metrics)
	print(f"[SAVED] {csv_path}")

	# save cache (for sweep)
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