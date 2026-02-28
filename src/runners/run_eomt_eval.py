import os
import json
import glob
import argparse
from datetime import datetime
from typing import Tuple, List, Dict, Any

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score

from src.models.eomt_wrapper import EoMTWrapper
from src.utils.ood_metrics import fpr_at_95_tpr


# ------------------------
# Dataset helpers
# ------------------------
def gt_path_from_image(path_img: str) -> str:
    """
    Derives the ground truth mask path from the input image path.
    Handles standard datasets like RoadAnomaly21 and LostAndFound.
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

    return root


def remap_ood_mask(path_gt: str, ood: np.ndarray) -> np.ndarray:
    """
    Remaps dataset-specific labels to a binary OOD format:
    0: In-Distribution, 1: Out-of-Distribution, 255: Ignore/Void.
    Follows the official evalAnomaly.py implementation provided in the course.
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
    """Loads and preprocesses the OOD ground truth mask."""
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
    mask_logits: torch.Tensor,   # [B, Q, h, w]
    class_logits: torch.Tensor,  # [B, Q, C(+1)]
    num_classes: int,
    temperature: float,
) -> torch.Tensor:
    """
    Performs MaskFormer-style pixel probability composition:
    1. Apply temperature scaling and Softmax to class logits.
    2. Apply Sigmoid to mask logits.
    3. Compute weighted sum via Einstein summation.
    4. Normalize across classes to ensure valid probability distributions.
    """
    if class_logits.shape[-1] == num_classes + 1:
        class_logits = class_logits[..., :num_classes]

    class_prob = F.softmax(class_logits / temperature, dim=-1)
    mask_prob = torch.sigmoid(mask_logits)

    pixel = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
    den = pixel.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return pixel / den


def anomaly_from_pixel_probs(pixel_probs: torch.Tensor, method: str) -> torch.Tensor:
    """Computes anomaly scores (higher = more OOD) using MSP, Entropy, or MaxLogit."""
    if method == "msp":
        msp = pixel_probs.max(dim=1).values
        return 1.0 - msp

    if method == "maxentropy":
        ent = -(pixel_probs * pixel_probs.clamp_min(1e-12).log()).sum(dim=1)
        return ent

    if method == "maxlogit":
        # Log-probability proxy for MaxLogit
        logp = pixel_probs.clamp_min(1e-12).log()
        m = logp.max(dim=1).values
        return -m

    raise ValueError(f"Unknown method: {method}")


def rba_from_masks(
    mask_logits: torch.Tensor,
    class_logits: torch.Tensor,
    num_classes: int,
    temperature: float,
    area_pow: float = 0.5,
) -> torch.Tensor:
    """
    Implements Region-based Anomaly (RbA) scoring.
    Computes region reliability by combining class confidence and mask area.
    """
    if class_logits.shape[-1] == num_classes + 1:
        class_logits = class_logits[..., :num_classes]

    class_prob = F.softmax(class_logits / temperature, dim=-1)
    conf = class_prob.max(dim=-1).values

    mask_prob = torch.sigmoid(mask_logits)
    area = mask_prob.mean(dim=(-2, -1))

    reliability = conf * (area.clamp_min(1e-6) ** area_pow)
    reliability = reliability.unsqueeze(-1).unsqueeze(-1)

    normality = (reliability * mask_prob).amax(dim=1)
    return 1.0 - normality


# ------------------------
# Artifacts & Logging
# ------------------------
def ensure_dirs(artifacts_dir: str) -> Tuple[str, str]:
    """Creates directory structure for saving results and logits."""
    res_dir = os.path.join(artifacts_dir, "results")
    log_dir = os.path.join(artifacts_dir, "logits")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return res_dir, log_dir


def append_metrics_csv(csv_path: str, row: Dict[str, Any]) -> None:
    """Appends evaluation metrics to a centralized CSV file."""
    header = [
        "timestamp", "dataset", "model", "method", "temperature",
        "auprc", "fpr95", "images_used", "resize_h", "resize_w",
        "num_q", "num_blocks", "backbone_name",
    ]
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in header) + "\n")


# ------------------------
# Execution Logic
# ------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Glob pattern for images")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    ap.add_argument("--dataset-name", required=True, help="Short name (e.g. RA21)")
    ap.add_argument("--method", choices=["msp", "maxlogit", "maxentropy", "rba"], default="msp")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-classes", type=int, default=19)
    ap.add_argument("--resize", default=None, help="Target HxW (e.g. 640x640)")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--save-logits", action="store_true", help="Enable raw logit caching for sweep")
    ap.add_argument("--strict-load", action="store_true", help="Enforce model/ckpt compatibility")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[device]", device)

    # Resolution inference
    if args.resize is not None:
        hw = args.resize.lower().replace(" ", "").split("x")
        H, W = int(hw[0]), int(hw[1])
    else:
        cfg_lower = os.path.basename(args.config).lower()
        H = W = 1024 if "1024" in cfg_lower else 640
    size_hw = (H, W)

    input_transform = Compose([Resize(size_hw, Image.BILINEAR), ToTensor()])

    model = EoMTWrapper(config_path=args.config, img_size=size_hw, num_classes=args.num_classes)
    model.load(args.ckpt, device, strict_load=args.strict_load)

    paths = sorted(glob.glob(os.path.expanduser(args.input)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found: {args.input}")

    res_dir, log_dir = ensure_dirs(args.artifacts_dir)
    metrics_csv = os.path.join(res_dir, "metrics.csv")

    anomaly_list, ood_list, names = [], [], []
    mask_cache, class_cache = [], []

    for p in paths:
        try:
            ood = load_ood_mask(p, size_hw=size_hw)
        except Exception as e:
            print(f"[SKIP] GT error {p}: {e}")
            continue

        if 1 not in np.unique(ood):
            continue

        img = Image.open(p).convert("RGB")
        x = input_transform(img).unsqueeze(0).to(device)

        mask_logits, class_logits = model.forward_masks_and_classes(x)

        if args.method == "rba":
            anomaly = rba_from_masks(mask_logits, class_logits, args.num_classes, args.temperature)
        else:
            pixel_probs = pixel_probs_from_masks(mask_logits, class_logits, args.num_classes, args.temperature)
            anomaly = anomaly_from_pixel_probs(pixel_probs, args.method)

        # Bilinear upsampling to match ground truth resolution
        if anomaly.shape[-2:] != size_hw:
            anomaly = F.interpolate(anomaly.unsqueeze(1), size=size_hw, mode="bilinear", align_corners=False).squeeze(1)

        anomaly_list.append(anomaly.squeeze(0).cpu().numpy())
        ood_list.append(ood)
        names.append(os.path.basename(p))

        if args.save_logits:
            # Cache raw FP16 logits for ultra-fast post-hoc temperature sweep
            mask_cache.append(mask_logits.squeeze(0).cpu().to(torch.float16).numpy())
            class_cache.append(class_logits.squeeze(0).cpu().to(torch.float16).numpy())

    # Metric computation using centralized OOD metrics
    ood_gts = np.array(ood_list)
    anomaly_scores = np.array(anomaly_list)
    val_out = np.concatenate([anomaly_scores[ood_gts == 0], anomaly_scores[ood_gts == 1]])
    val_label = np.concatenate([np.zeros((ood_gts == 0).sum()), np.ones((ood_gts == 1).sum())])

    auprc = float(average_precision_score(val_label, val_out))
    fpr95 = float(fpr_at_95_tpr(val_out, val_label))

    print(f"Results: AUPRC {auprc*100:.2f} | FPR95 {fpr95*100:.2f}")

    # Persistence
    stamp = datetime.utcnow().isoformat() + "Z"
    meta = getattr(model, "meta", {})
    row = {
        "timestamp": stamp, "dataset": args.dataset_name, "model": "EoMT",
        "method": args.method, "temperature": args.temperature, "auprc": auprc,
        "fpr95": fpr95, "images_used": len(anomaly_list), "resize_h": H, "resize_w": W,
        "num_q": meta.get("num_q", ""), "num_blocks": meta.get("num_blocks", ""),
        "backbone_name": meta.get("backbone_name", ""),
    }
    append_metrics_csv(metrics_csv, row)

    if args.save_logits:
        np.save(os.path.join(log_dir, f"{args.dataset_name}__mask_logits_f16.npy"), np.array(mask_cache))
        np.save(os.path.join(log_dir, f"{args.dataset_name}__class_logits_f16.npy"), np.array(class_cache))
        np.save(os.path.join(log_dir, f"{args.dataset_name}__gt.npy"), ood_gts.astype(np.uint8))
        with open(os.path.join(log_dir, f"{args.dataset_name}__names.json"), "w") as f:
            json.dump(names, f)

if __name__ == "__main__":
    main()