# src/eval/miou_cityscapes.py
#
# Evaluates EoMT on the Cityscapes validation set (500 images, 19 classes).
# Computes mean Intersection over Union (mIoU) and per-class IoU.
#
# Supports three checkpoints:
#   - Cityscapes-trained  (eomt_cityscapes.bin)  num_classes=19
#   - COCO-trained        (eomt_coco.bin)         num_classes=133, needs remapping
#   - Fine-tuned          (eomt_finetuned.bin)    num_classes=19
#
# COCO -> Cityscapes class remapping strategy:
#   Only COCO classes that have a direct semantic equivalent in Cityscapes
#   are mapped. Pixels whose predicted COCO class has no Cityscapes equivalent
#   are treated as void and excluded from the IoU computation.
#   This follows the standard zero-shot cross-dataset evaluation protocol.
#
# Usage:
#   python3 -m src.eval.miou_cityscapes \
#       --images-dir  /content/cityscapes/leftImg8bit/val \
#       --gt-dir      /content/cityscapes/gtFine/val \
#       --ckpt        /content/drive/MyDrive/anom_project/ckpts/eomt/eomt_cityscapes.bin \
#       --config      eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
#       --mode        cityscapes \
#       --output-json /content/drive/MyDrive/anom_project/results/miou_cityscapes.json

import os
import glob
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor

from src.models.eomt_wrapper import EoMTWrapper
from src.utils.determinism import apply_determinism


# Cityscapes 19-class label mapping
# Maps raw gtFine label IDs to train IDs (0-18), 255 = ignore
CITYSCAPES_LABEL_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255,
    6: 255, 7: 0,   8: 1,   9: 255, 10: 255, 11: 2,
    12: 3,  13: 4,  14: 255, 15: 255, 16: 255, 17: 5,
    18: 255, 19: 6,  20: 7,  21: 8,  22: 9,  23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
    30: 255, 31: 16, 32: 17, 33: 18,
}

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

# COCO panoptic -> Cityscapes train ID remapping
# Only classes with a clear semantic equivalent are mapped.
# All others map to 255 (void / ignore).
COCO_TO_CITYSCAPES_TRAINID = {
    # road
    148: 0,   # pavement-merged
    119: 0,   # road
    # sidewalk
    11: 1,    # pavement
    # building
    13: 2,    # building-other-merged
    # wall
    175: 3,   # wall-other-merged
    # fence
    33: 4,    # fence
    # vegetation
    96: 8,    # tree-merged
    # sky
    156: 10,  # sky-other-merged
    # person
    1: 11,    # person
    # car
    3: 13,    # car
    # truck
    8: 14,    # truck
    # bus
    6: 15,    # bus
    # train
    7: 16,    # train
    # motorcycle
    4: 17,    # motorcycle
    # bicycle
    2: 18,    # bicycle
}

NUM_CLASSES = 19
IGNORE_INDEX = 255


def remap_gt(gt_path: str) -> np.ndarray:
    """
    Loads a Cityscapes gtFine_labelIds PNG and remaps to train IDs (0-18).
    Pixels with no valid train ID are set to IGNORE_INDEX (255).
    """
    raw = np.array(Image.open(gt_path), dtype=np.int32)
    out = np.full_like(raw, IGNORE_INDEX, dtype=np.int32)
    for label_id, train_id in CITYSCAPES_LABEL_TO_TRAINID.items():
        out[raw == label_id] = train_id
    return out


def predict_cityscapes(
    model: EoMTWrapper,
    img_pil: Image.Image,
    size_hw: tuple,
    device: torch.device,
) -> np.ndarray:
    """
    Runs EoMT forward pass and returns predicted class map [H, W] at original resolution.
    """
    orig_h, orig_w = img_pil.height, img_pil.width
    transform = Compose([Resize(size_hw, Image.BILINEAR), ToTensor()])
    x = transform(img_pil).unsqueeze(0).float().to(device)

    with torch.no_grad():
        mask_logits, class_logits = model.forward_masks_and_classes(x)

    # pixel logits [1, C, h, w]
    cl = class_logits[..., :-1]  # drop no-object
    pixel_logits = torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        cl.softmax(dim=-1),
    )

    # upsample to original resolution
    pixel_logits = F.interpolate(
        pixel_logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )

    pred = pixel_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    return pred


def predict_coco(
    model: EoMTWrapper,
    img_pil: Image.Image,
    size_hw: tuple,
    device: torch.device,
) -> np.ndarray:
    """
    Runs EoMT COCO forward pass and remaps predicted COCO class IDs
    to Cityscapes train IDs. Unmapped classes become IGNORE_INDEX.
    """
    orig_h, orig_w = img_pil.height, img_pil.width
    transform = Compose([Resize(size_hw, Image.BILINEAR), ToTensor()])
    x = transform(img_pil).unsqueeze(0).float().to(device)

    with torch.no_grad():
        mask_logits, class_logits = model.forward_masks_and_classes(x)

    cl = class_logits[..., :-1]
    pixel_logits = torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        cl.softmax(dim=-1),
    )
    pixel_logits = F.interpolate(
        pixel_logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )

    coco_pred = pixel_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    # remap COCO class IDs to Cityscapes train IDs
    out = np.full_like(coco_pred, IGNORE_INDEX)
    for coco_id, cs_id in COCO_TO_CITYSCAPES_TRAINID.items():
        out[coco_pred == coco_id] = cs_id

    return out


class IoUMeter:
    """Accumulates confusion matrix for mIoU computation."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.conf_matrix  = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        mask = gt != self.ignore_index
        p = pred[mask].astype(np.int64)
        g = gt[mask].astype(np.int64)
        # Clamp predictions to valid range
        p = np.clip(p, 0, self.num_classes - 1)
        np.add.at(self.conf_matrix, (g, p), 1)

    def compute(self) -> dict:
        tp  = np.diag(self.conf_matrix)
        fp  = self.conf_matrix.sum(axis=0) - tp
        fn  = self.conf_matrix.sum(axis=1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1e-6)
        miou = float(np.mean(iou))
        return {
            "miou":          miou,
            "miou_pct":      miou * 100.0,
            "per_class_iou": {
                CITYSCAPES_CLASS_NAMES[i]: float(iou[i]) * 100.0
                for i in range(self.num_classes)
            },
        }


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate EoMT mIoU on Cityscapes validation set."
    )
    ap.add_argument("--images-dir",  required=True,
                    help="Path to leftImg8bit/val")
    ap.add_argument("--gt-dir",      required=True,
                    help="Path to gtFine/val")
    ap.add_argument("--ckpt",        required=True)
    ap.add_argument("--config",      required=True)
    ap.add_argument("--mode",        choices=["cityscapes", "coco", "finetuned"],
                    required=True,
                    help="cityscapes/finetuned: direct 19-class prediction. "
                         "coco: COCO->Cityscapes remapping applied.")
    ap.add_argument("--num-classes", type=int, default=19,
                    help="19 for cityscapes/finetuned, 133 for coco")
    ap.add_argument("--resize",      default="640x640")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--cpu",         action="store_true")
    args = ap.parse_args()

    apply_determinism(mode="robust", seed=args.seed, deterministic=True)

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")

    h, w    = [int(x) for x in args.resize.lower().split("x")]
    size_hw = (h, w)

    backbone = (
        "vit_large_patch14_reg4_dinov2"
        if "large" in os.path.basename(args.config).lower()
        else "vit_base_patch14_reg4_dinov2"
    )

    model = EoMTWrapper(
        img_size=size_hw,
        num_classes=args.num_classes,
        num_q=100,
        num_blocks=3,
        backbone_name=backbone,
        masked_attn_enabled=True,
    )
    model.load(args.ckpt, device)
    print(f"[ckpt] {args.ckpt}")

    # Gather image paths
    img_paths = sorted(glob.glob(
        os.path.join(args.images_dir, "*", "*_leftImg8bit.png")
    ))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")
    print(f"[images] {len(img_paths)} found")

    meter = IoUMeter(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)

    for i, img_path in enumerate(img_paths):
        img_pil = Image.open(img_path).convert("RGB")

        # Build GT path from image path
        # leftImg8bit/val/<city>/<city>_<id>_leftImg8bit.png
        # gtFine/val/<city>/<city>_<id>_gtFine_labelIds.png
        parts   = Path(img_path)
        city    = parts.parent.name
        stem    = parts.stem.replace("_leftImg8bit", "")
        gt_path = os.path.join(
            args.gt_dir, city, f"{stem}_gtFine_labelIds.png"
        )

        if not os.path.exists(gt_path):
            print(f"[SKIP] GT not found: {gt_path}")
            continue

        gt = remap_gt(gt_path)

        if args.mode in ("cityscapes", "finetuned"):
            pred = predict_cityscapes(model, img_pil, size_hw, device)
        else:
            pred = predict_coco(model, img_pil, size_hw, device)

        meter.update(pred, gt)

        if (i + 1) % 50 == 0:
            interim = meter.compute()
            print(f"[{i+1}/{len(img_paths)}] mIoU so far: {interim['miou_pct']:.2f}%")

    results = meter.compute()
    results["ckpt"]        = args.ckpt
    results["mode"]        = args.mode
    results["num_classes"] = args.num_classes
    results["resize"]      = args.resize
    results["num_images"]  = len(img_paths)

    print(f"\nmIoU: {results['miou_pct']:.4f}%")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {args.output_json}")


if __name__ == "__main__":
    main()
