import os
from typing import Tuple

import numpy as np
from PIL import Image

from torchvision.transforms import Resize


def gt_path_from_image(path_img: str) -> str:
	"""
	Matches professor convention: swap 'images' -> 'labels_masks'
	and force .png for known datasets.
	"""
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

	# fallback: keep extension if present, else add .png
	ext = os.path.splitext(root)[1].lower()
	if ext:
		return root
	return root + ".png"


def remap_ood_mask(path_gt: str, ood: np.ndarray) -> np.ndarray:
	"""
	Remapping exactly in-line with the professor-style scripts you pasted.
	Output convention:
	- 1 = OOD
	- 0 = InD
	(other values may exist in raw GT but should be remapped away)
	"""
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


def has_ood_pixels(ood: np.ndarray) -> bool:
	return bool((ood == 1).any())