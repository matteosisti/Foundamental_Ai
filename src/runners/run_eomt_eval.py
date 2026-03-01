# src/runners/run_eomt_eval.py
import os
import glob
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score

from src.models.eomt_wrapper import EoMTWrapper
from src.utils.artifacts import create_run_dir
from src.utils.ood_metrics import fpr_at_95_tpr
from src.utils.determinism import apply_determinism


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

	return root


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


def load_ood_mask(path_img: str, size_hw: Tuple[int, int]) -> np.ndarray:
	path_gt = gt_path_from_image(path_img)
	mask = Image.open(path_gt)
	mask = Resize(size_hw, Image.NEAREST)(mask)
	ood = np.array(mask)
	ood = remap_ood_mask(path_gt, ood)
	return ood


def pixel_probs_from_masks(
	mask_logits: torch.Tensor,	# [B,Q,h,w]
	class_logits: torch.Tensor,	# [B,Q,C(+1)]
	num_classes: int,
	temperature: float,
) -> torch.Tensor:
	if class_logits.shape[-1] == num_classes + 1:
		class_logits = class_logits[..., :num_classes]

	class_prob = F.softmax(class_logits / temperature, dim=-1)	# [B,Q,C]
	mask_prob = torch.sigmoid(mask_logits)						# [B,Q,h,w]

	pixel = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)	# [B,C,h,w]
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
		# proxy: log(prob)
		logits_proxy = pixel_probs.clamp_min(1e-12).log()
		m = logits_proxy.max(dim=1).values
		return -m

	raise ValueError(f"Unknown method: {method}")


def rba_from_masks(
	mask_logits: torch.Tensor,	# [B,Q,h,w]
	class_logits: torch.Tensor,	# [B,Q,C(+1)]
	num_classes: int,
	temperature: float,
	area_pow: float = 0.5,
) -> torch.Tensor:
	if class_logits.shape[-1] == num_classes + 1:
		class_logits = class_logits[..., :num_classes]

	class_prob = F.softmax(class_logits / temperature, dim=-1)	# [B,Q,C]
	conf = class_prob.max(dim=-1).values						# [B,Q]

	mask_prob = torch.sigmoid(mask_logits)						# [B,Q,h,w]
	area = mask_prob.mean(dim=(-2, -1))						# [B,Q]

	reliability = conf * (area.clamp_min(1e-6) ** area_pow)		# [B,Q]
	reliability = reliability.unsqueeze(-1).unsqueeze(-1)		# [B,Q,1,1]

	normality = (reliability * mask_prob).amax(dim=1)			# [B,h,w]
	anomaly = 1.0 - normality
	return anomaly


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

	ap.add_argument("--input", required=True, help="Glob images, e.g. .../images/*.*")
	ap.add_argument("--ckpt", required=True, help="EoMT checkpoint (.bin)")
	ap.add_argument("--config", default="eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml")

	ap.add_argument("--dataset-name", required=True, help="Short name: RA21, RO21, FS_STATIC, LAF, ...")

	ap.add_argument("--method", choices=["msp", "maxlogit", "maxentropy", "rba"], default="msp")
	ap.add_argument("--temperature", type=float, default=1.0)
	ap.add_argument("--num-classes", type=int, default=19)
	ap.add_argument("--resize", default=None, help="HxW e.g. 640x640. If None inferred from config filename.")
	ap.add_argument("--mode", choices=["robust", "prof-exact"], default="robust")

	ap.add_argument("--seed", type=int, default=0)
	ap.add_argument("--deterministic", action="store_true", help="Force determinism even in prof-exact")

	ap.add_argument("--artifacts-dir", default="artifacts")
	ap.add_argument("--save-logits", action="store_true", help="Cache raw mask/class logits + gt + names for sweep")
	ap.add_argument("--cpu", action="store_true")

	args = ap.parse_args()

	apply_determinism(mode=args.mode, seed=args.seed, deterministic=args.deterministic)

	device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
	print("[device]", device)

	# infer resize
	if args.resize is None:
		cfg_lower = os.path.basename(args.config).lower()
		if "1024" in cfg_lower:
			H = W = 1024
		elif "640" in cfg_lower:
			H = W = 640
		else:
			H = W = 640
	else:
		hw = args.resize.lower().replace(" ", "").split("x")
		if len(hw) != 2:
			raise ValueError("--resize must be like 640x640")
		H, W = int(hw[0]), int(hw[1])

	size_hw = (H, W)

	input_transform = Compose([
		Resize(size_hw, Image.BILINEAR),
		ToTensor(),
	])

	# create run dir automatically
	art = create_run_dir(
		artifacts_root=args.artifacts_dir,
		dataset=args.dataset_name,
		model="EoMT",
		method=args.method,
		temperature=args.temperature,
		mode=args.mode,
		extra={
			"ckpt": args.ckpt,
			"config": args.config,
			"input_glob": args.input,
			"resize_h": H,
			"resize_w": W,
			"num_classes": args.num_classes,
			"seed": args.seed,
			"deterministic": bool(args.deterministic),
		},
	)
	print("[ARTIFACTS]", art.root)

	# init model (same assumptions)
	backbone = "vit_base_patch14_reg4_dinov2"
	num_q = 100
	num_blocks = 3
	if "large" in os.path.basename(args.config).lower():
		backbone = "vit_large_patch14_reg4_dinov2"

	model = EoMTWrapper(
		img_size=size_hw,
		num_classes=args.num_classes,
		num_q=num_q,
		num_blocks=num_blocks,
		backbone_name=backbone,
		masked_attn_enabled=True,
	)
	model.load(args.ckpt, device)

	paths = sorted(glob.glob(os.path.expanduser(args.input)))
	if len(paths) == 0:
		raise FileNotFoundError(f"No images found for: {args.input}")

	anomaly_list: List[np.ndarray] = []
	ood_list: List[np.ndarray] = []
	names: List[str] = []

	mask_logits_cache: List[np.ndarray] = []
	class_logits_cache: List[np.ndarray] = []

	for p in paths:
		try:
			ood = load_ood_mask(p, size_hw=size_hw)
		except Exception as e:
			print(f"[SKIP] GT error {p}: {e}")
			continue

		if 1 not in np.unique(ood):
			continue

		img = Image.open(p).convert("RGB")
		x = input_transform(img).unsqueeze(0).float().to(device)

		mask_logits, class_logits = model.forward_masks_and_classes(x)  # [1,Q,h,w], [1,Q,C(+1)]

		if args.method == "rba":
			anomaly = rba_from_masks(
				mask_logits=mask_logits,
				class_logits=class_logits,
				num_classes=args.num_classes,
				temperature=args.temperature,
			)  # [1,h,w]
		else:
			pixel_probs = pixel_probs_from_masks(
				mask_logits=mask_logits,
				class_logits=class_logits,
				num_classes=args.num_classes,
				temperature=args.temperature,
			)  # [1,C,h,w]
			anomaly = anomaly_from_pixel_probs(pixel_probs, args.method)  # [1,h,w]

		# upsample anomaly to input size if needed
		if anomaly.shape[-2:] != size_hw:
			anomaly = F.interpolate(
				anomaly.unsqueeze(1),
				size=size_hw,
				mode="bilinear",
				align_corners=False,
			).squeeze(1)

		anomaly_np = anomaly.squeeze(0).detach().cpu().float().numpy()
		anomaly_list.append(anomaly_np)
		ood_list.append(ood)
		names.append(os.path.basename(p))

		if args.save_logits:
			# cache raw logits (temperature-agnostic)
			mask_logits_cache.append(mask_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())
			class_logits_cache.append(class_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())

	n_used = len(anomaly_list)
	if n_used == 0:
		raise RuntimeError("No valid images used (maybe mapping or empty OOD regions).")

	ood_gts = np.array(ood_list)
	anomaly_scores = np.array(anomaly_list)

	ood_mask = (ood_gts == 1)
	in_mask = (ood_gts == 0)

	ood_out = anomaly_scores[ood_mask]
	in_out = anomaly_scores[in_mask]

	val_out = np.concatenate([in_out, ood_out])
	val_label = np.concatenate([np.zeros(len(in_out)), np.ones(len(ood_out))])

	auprc = float(average_precision_score(val_label, val_out))
	fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

	metrics = {
		"timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
		"model": "EoMT",
		"dataset": args.dataset_name,
		"method": args.method,
		"temperature": float(args.temperature),
		"mode": args.mode,
		"num_classes": int(args.num_classes),
		"resize_h": int(H),
		"resize_w": int(W),
		"ckpt": args.ckpt,
		"config": args.config,
		"seed": int(args.seed),
		"deterministic": bool(args.deterministic),
		"auprc": auprc,
		"fpr95": fpr95,
		"auprc_pct": auprc * 100.0,
		"fpr95_pct": fpr95 * 100.0,
		"images_used": int(n_used),
	}

	print("=====================================")
	print(f"EoMT | dataset={args.dataset_name} | method={args.method} | T={args.temperature} | mode={args.mode}")
	print(f"AUPRC: {metrics['auprc_pct']:.4f}")
	print(f"FPR@95TPR: {metrics['fpr95_pct']:.4f}")
	print(f"Images used: {metrics['images_used']}")
	print(f"Resize: {H}x{W}")
	print("=====================================")

	json_path = art.results / "metrics.json"
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	print(f"[SAVED] {json_path}")

	csv_path = art.results / "metrics.csv"
	append_metrics_csv(csv_path, metrics)
	print(f"[SAVED] {csv_path}")

	# cache for sweep
	if args.save_logits:
		np.save(art.logits / f"{args.dataset_name}__mask_logits_f16.npy", np.array(mask_logits_cache, dtype=np.float16))
		np.save(art.logits / f"{args.dataset_name}__class_logits_f16.npy", np.array(class_logits_cache, dtype=np.float16))
		np.save(art.logits / f"{args.dataset_name}__gt.npy", ood_gts.astype(np.uint8))
		with open(art.logits / f"{args.dataset_name}__names.json", "w", encoding="utf-8") as f:
			json.dump(names, f, indent=2)

		print(f"[CACHED] {art.logits / f'{args.dataset_name}__mask_logits_f16.npy'}")
		print(f"[CACHED] {art.logits / f'{args.dataset_name}__class_logits_f16.npy'}")
		print(f"[CACHED] {art.logits / f'{args.dataset_name}__gt.npy'}")
		print(f"[CACHED] {art.logits / f'{args.dataset_name}__names.json'}")


if __name__ == "__main__":
	main()