import os
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor

from src.models.erfnet import ERFNet
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score

NUM_CLASSES = 20

input_transform = Compose([
	Resize((512, 1024), Image.BILINEAR),
	ToTensor(),
])

target_transform = Compose([
	Resize((512, 1024), Image.NEAREST),
])

def load_erfnet(weights_path: str, device: torch.device) -> torch.nn.Module:
	model = ERFNet(NUM_CLASSES)

	if device.type == "cuda":
		model = torch.nn.DataParallel(model).cuda()
	else:
		model = model.cpu()

	state = torch.load(weights_path, map_location="cpu")
	own = model.state_dict()

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
	ext = os.path.splitext(root)[1].lower()

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

@torch.no_grad()
def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--input", required=True, help="Glob, es: /path/images/*.*")
	ap.add_argument("--weights", required=True, help="Path .pth ERFNet")
	ap.add_argument("--cpu", action="store_true")
	ap.add_argument("--method", choices=["msp","maxlogit","maxentropy"], default="msp")
	ap.add_argument("--temperature", type=float, default=1.0)
	args = ap.parse_args()

	device = torch.device("cpu" if args.cpu else "cuda")

	model = load_erfnet(args.weights, device)

	paths = sorted(glob.glob(os.path.expanduser(args.input)))
	if len(paths) == 0:
		raise FileNotFoundError(f"Nessuna immagine trovata: {args.input}")

	anomaly_list = []
	ood_list = []

	for path in paths:
		try:
			ood = load_ood_mask(path)
		except Exception as e:
			print(f"[SKIP] GT error {path}: {e}")
			continue

		if 1 not in np.unique(ood):
			continue

		img = Image.open(path).convert("RGB")
		x = input_transform(img).unsqueeze(0).float().to(device)

		logits = model(x)
		anomaly = anomaly_from_logits(logits, args.method, args.temperature)

		ood_list.append(ood)
		anomaly_list.append(anomaly)

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

	print("=====================================")
	print(f"ERFNet | method={args.method} | T={args.temperature}")
	print(f"AUPRC: {auprc*100.0:.4f}")
	print(f"FPR@95TPR: {fpr95*100.0:.4f}")
	print("=====================================")

if __name__ == "__main__":
	main()