
import os
import json
import glob
import argparse
from datetime import datetime
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor

from sklearn.metrics import average_precision_score

from src.models.eomt_wrapper import EoMTWrapper


# ------------------------
# Metrics utils (no deps)
# ------------------------
def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
	"""
	FPR@95TPR for binary labels (0 in, 1 ood).
	scores: higher => more OOD
	"""
	scores = scores.astype(np.float64)
	labels = labels.astype(np.int64)

	ood = scores[labels == 1]
	ind = scores[labels == 0]
	if len(ood) == 0 or len(ind) == 0:
		return float("nan")

	# threshold at 95% TPR on OOD
	th = np.quantile(ood, 0.05)  # keep 95% of ood above threshold
	fpr = (ind >= th).mean()
	return float(fpr)


# ------------------------
# Dataset helpers
# ------------------------
def gt_path_from_image(path_img: str) -> str:
	path_gt = path_img.replace("images", "labels_masks")
	root = path_gt

	# normalize GT extension to .png (common in provided validation datasets)
	if "RoadObstacle21" in root or "RoadObsticle21" in root:
		return os.path.splitext(root)[0] + ".png"
	if "fs_static" in root:
		return os.path.splitext(root)[0] + ".png"
	if "RoadAnomaly21" in root or "RoadAnomaly" in root:
		return os.path.splitext(root)[0] + ".png"
	if "LostAndFound" in root or "FS_LostFound_full" in root:
		return os.path.splitext(root)[0] + ".png"
	return path_gt


def remap_ood_mask(path_gt: str, ood: np.ndarray) -> np.ndarray:
	# same remapping used in professor evalAnomaly.py
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


def load_ood_mask(path_img: str, size_hw: Tuple[int, int]) -> np.ndarray:
	path_gt = gt_path_from_image(path_img)
	mask = Image.open(path_gt)
	mask = Resize(size_hw, Image.NEAREST)(mask)
	ood = np.array(mask)
	ood = remap_ood_mask(path_gt, ood)
	return ood


# ------------------------
# EoMT -> pixel probabilities
# ------------------------
def pixel_probs_from_masks(
	mask_logits: torch.Tensor,	# [B,Q,H,W]
	class_logits: torch.Tensor,	# [B,Q,C(+1)]
	num_classes: int,
	temperature: float,
) -> torch.Tensor:
	"""
	MaskFormer-style composition:
	- class_prob = softmax(class_logits/T) over classes
	- mask_prob = sigmoid(mask_logits)
	- pixel_prob[c,h,w] = sum_q class_prob[q,c] * mask_prob[q,h,w]

	Then renormalize across classes to get a proper distribution per pixel.
	"""
	# drop "no-object" if present as last channel
	if class_logits.shape[-1] == num_classes + 1:
		class_logits = class_logits[..., :num_classes]

	class_prob = F.softmax(class_logits / temperature, dim=-1)	# [B,Q,C]
	mask_prob = torch.sigmoid(mask_logits)						# [B,Q,H,W]

	# compose: [B,C,H,W]
	pixel = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)

	# normalize to sum=1 per pixel
	den = pixel.sum(dim=1, keepdim=True).clamp_min(1e-8)
	pixel = pixel / den
	return pixel


def anomaly_from_pixel_probs(pixel_probs: torch.Tensor, method: str) -> torch.Tensor:
	"""
	pixel_probs: [B,C,H,W], normalized
	return anomaly score [B,H,W] (higher => more OOD)
	"""
	if method == "msp":
		msp = pixel_probs.max(dim=1).values
		return 1.0 - msp

	if method == "maxentropy":
		ent = -(pixel_probs * pixel_probs.clamp_min(1e-12).log()).sum(dim=1)
		return ent

	if method == "maxlogit":
		# No direct pixel logits; use log(prob) proxy.
		logits_proxy = pixel_probs.clamp_min(1e-12).log()
		m = logits_proxy.max(dim=1).values
		return -m

	raise ValueError(f"Unknown method: {method}")


# ------------------------
# Artifact saving
# ------------------------
def ensure_dirs(artifacts_dir: str) -> Tuple[str, str]:
	res_dir = os.path.join(artifacts_dir, "results")
	log_dir = os.path.join(artifacts_dir, "logits")
	os.makedirs(res_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	return res_dir, log_dir


def append_metrics_csv(csv_path: str, row: dict) -> None:
	header = [
		"timestamp",
		"dataset",
		"model",
		"method",
		"temperature",
		"auprc",
		"fpr95",
		"images_used",
		"resize_h",
		"resize_w",
	]
	is_new = not os.path.exists(csv_path)
	with open(csv_path, "a", encoding="utf-8") as f:
		if is_new:
			f.write(",".join(header) + "\n")
		f.write(",".join(str(row.get(k, "")) for k in header) + "\n")


# ------------------------
# Main
# ------------------------
@torch.no_grad()
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--input", required=True, help="Glob images, e.g. .../images/*.*")
	ap.add_argument("--ckpt", required=True, help="EoMT checkpoint (.bin)")
	ap.add_argument("--config", default="eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml")
	ap.add_argument("--dataset-name", required=True, help="Short name: RA21, RO21, FS_STATIC, LAF, RA")
	ap.add_argument("--method", choices=["msp", "maxlogit", "maxentropy"], default="msp")
	ap.add_argument("--temperature", type=float, default=1.0)
	ap.add_argument("--num-classes", type=int, default=19)
	ap.add_argument("--resize", default=None, help="HxW e.g. 640x640. If None inferred from config name.")
	ap.add_argument("--artifacts-dir", default="artifacts")
	ap.add_argument("--save-logits", action="store_true", help="Cache pixel_probs (float16) + gt + names")
	ap.add_argument("--cpu", action="store_true")
	args = ap.parse_args()

	# device
	device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
	print("[device]", device)

	# infer resize from config if not provided
	if args.resize is None:
		cfg_lower = os.path.basename(args.config).lower()
		if "1024" in cfg_lower:
			H = W = 1024
		elif "640" in cfg_lower:
			H = W = 640
		else:
			H = W = 640
	else:
		# parse "HxW"
		hw = args.resize.lower().replace(" ", "").split("x")
		if len(hw) != 2:
			raise ValueError("--resize must be like 640x640")
		H, W = int(hw[0]), int(hw[1])

	size_hw = (H, W)

	# transforms
	input_transform = Compose([
		Resize(size_hw, Image.BILINEAR),
		ToTensor(),
	])

	# init model (we read minimal fields from YAML name)
	# For now we assume base_640 => vit_base_patch14_reg4_dinov2, num_q=100, num_blocks=3
	# You can extend later by parsing YAML if needed.
	backbone = "vit_base_patch14_reg4_dinov2"
	num_q = 100
	num_blocks = 3

	if "large" in os.path.basename(args.config).lower():
		# most likely large model
		backbone = "vit_large_patch14_reg4_dinov2"
		# still keep defaults unless you confirm config differs
		num_q = 100
		num_blocks = 3

	model = EoMTWrapper(
		img_size=size_hw,
        num_classes=args.num_classes,
		num_q=num_q,
		num_blocks=num_blocks,
		backbone_name=backbone,
		masked_attn_enabled=True,
	)
	model.load(args.ckpt, device)

	# paths
	paths = sorted(glob.glob(os.path.expanduser(args.input)))
	if len(paths) == 0:
		raise FileNotFoundError(f"No images found for: {args.input}")

	# artifacts
	res_dir, log_dir = ensure_dirs(args.artifacts_dir)
	metrics_csv = os.path.join(res_dir, "metrics.csv")

	# run
	anomaly_list: List[np.ndarray] = []
	ood_list: List[np.ndarray] = []
	names: List[str] = []
	pixel_cache: List[np.ndarray] = []

	for p in paths:
		try:
			ood = load_ood_mask(p, size_hw=size_hw)
		except Exception as e:
			print(f"[SKIP] GT error {p}: {e}")
			continue

		# keep only images that contain OOD pixels
		if 1 not in np.unique(ood):
			continue

		img = Image.open(p).convert("RGB")
		x = input_transform(img).unsqueeze(0).float().to(device)	# [1,3,H,W]

		mask_logits, class_logits = model.forward_masks_and_classes(x)

		pixel_probs = pixel_probs_from_masks(
			mask_logits=mask_logits,
			class_logits=class_logits,
			num_classes=args.num_classes,
			temperature=args.temperature,
		)  # [1,C,H,W]

		anomaly = anomaly_from_pixel_probs(pixel_probs, args.method)  # [1,H,W]
		if anomaly.shape[-2:] != size_hw:
			anomaly = F.interpolate(
				anomaly.unsqueeze(1),  # [1,1,h,w]
                size=size_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # [1,H,W]
		anomaly_np = anomaly.squeeze(0).detach().cpu().float().numpy()

		anomaly_list.append(anomaly_np)
		ood_list.append(ood)
		names.append(os.path.basename(p))

		if args.save_logits:
			# cache pixel probs, float16 to reduce size
			pixel_cache.append(pixel_probs.squeeze(0).detach().cpu().to(torch.float16).numpy())

	# metrics
	ood_gts = np.array(ood_list)				# [N,H,W]
	anomaly_scores = np.array(anomaly_list)		# [N,H,W]
	n_used = len(anomaly_list)

	if n_used == 0:
		raise RuntimeError("No valid images used (maybe GT mapping or empty OOD regions).")

	ood_mask = (ood_gts == 1)
	in_mask = (ood_gts == 0)

	ood_out = anomaly_scores[ood_mask]
	in_out = anomaly_scores[in_mask]

	val_out = np.concatenate([in_out, ood_out])
	val_label = np.concatenate([np.zeros(len(in_out)), np.ones(len(ood_out))])

	auprc = float(average_precision_score(val_label, val_out))
	fpr95 = float(fpr_at_95_tpr(val_out, val_label))

	print("=====================================")
	print(f"EoMT | dataset={args.dataset_name} | method={args.method} | T={args.temperature}")
	print(f"AUPRC: {auprc*100.0:.4f}")
	print(f"FPR@95TPR: {fpr95*100.0:.4f}")
	print(f"Images used: {n_used}")
	print(f"Resize: {H}x{W}")
	print("=====================================")

	# save json result
	stamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
	out = {
		"timestamp_utc": stamp,
		"dataset": args.dataset_name,
		"model": "EoMT",
		"config": args.config,
		"ckpt": args.ckpt,
		"method": args.method,
		"temperature": args.temperature,
		"num_classes": args.num_classes,
		"resize_h": H,
		"resize_w": W,
		"auprc": auprc,
		"fpr95": fpr95,
		"images_used": n_used,
	}

	json_name = f"{args.dataset_name}__EoMT__{args.method}__T{args.temperature}.json"
	json_path = os.path.join(res_dir, json_name)
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(out, f, indent=2)
	print(f"[SAVED] {json_path}")

	# append metrics.csv
	append_metrics_csv(metrics_csv, {
		"timestamp": stamp,
		"dataset": args.dataset_name,
		"model": "EoMT",
		"method": args.method,
		"temperature": args.temperature,
		"auprc": auprc,
		"fpr95": fpr95,
		"images_used": n_used,
		"resize_h": H,
		"resize_w": W,
	})
	print(f"[UPDATED] {metrics_csv}")

	# cache
	if args.save_logits:
		logits_path = os.path.join(log_dir, f"{args.dataset_name}__pixel_probs_f16.npy")
		gt_path = os.path.join(log_dir, f"{args.dataset_name}__gt.npy")
		names_path = os.path.join(log_dir, f"{args.dataset_name}__names.json")

		np.save(logits_path, np.array(pixel_cache, dtype=np.float16))
		np.save(gt_path, ood_gts.astype(np.uint8))
		with open(names_path, "w", encoding="utf-8") as f:
			json.dump(names, f, indent=2)

		print(f"[CACHED] {logits_path}")
		print(f"[CACHED] {gt_path}")
		print(f"[CACHED] {names_path}")


if __name__ == "__main__":
	main()
