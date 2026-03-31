# src/runners/run_eomt_eval.py

import os
import glob
import json
import csv
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List, Dict, Any

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
from src.utils.ood_dataset import gt_path_from_image, load_ood_mask
from src.utils.eomt_post import (
    pixel_probs_from_masks,
    anomaly_from_pixel_probs,
    anomaly_maxlogit_from_masks,
    rba_from_masks,
)


def _sha1_8_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


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

    ap.add_argument("--input",        required=True)
    ap.add_argument("--ckpt",         required=True)
    ap.add_argument("--config",       default="eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml")
    ap.add_argument("--dataset-name", required=True)

    ap.add_argument("--method",      choices=["msp", "maxlogit", "maxentropy", "rba"], default="msp")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-classes", type=int,   default=19)
    ap.add_argument("--resize",      default=None,
                    help="Override input resolution, e.g. 1024x1024")
    ap.add_argument("--mode",        choices=["robust", "prof-exact"], default="robust")

    ap.add_argument("--seed",          type=int,  default=0)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--save-logits",   action="store_true")
    ap.add_argument("--cpu",           action="store_true")

    args = ap.parse_args()

    # Determinism policy
    want_determinism = (args.mode == "robust") or bool(args.deterministic)
    apply_determinism(mode=args.mode, seed=int(args.seed), deterministic=bool(want_determinism))

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[device]", device)

    # Resolve input resolution
    if args.resize is not None:
        hw = args.resize.lower().replace(" ", "").split("x")
        if len(hw) != 2:
            raise ValueError("--resize must be formatted as HxW, e.g. 640x640")
        H, W = int(hw[0]), int(hw[1])
    else:
        cfg_lower = os.path.basename(args.config).lower()
        H = W = 1024 if "1024" in cfg_lower else 640

    size_hw = (H, W)

    input_transform = Compose([
        Resize(size_hw, Image.BILINEAR),
        ToTensor(),
    ])

    ckpt_basename = os.path.basename(args.ckpt)
    ckpt_sha1_8   = _sha1_8_of_file(args.ckpt)

    art = create_run_dir(
        artifacts_root=args.artifacts_dir,
        dataset=args.dataset_name,
        model="EoMT",
        method=args.method,
        temperature=args.temperature,
        mode=args.mode,
        extra={
            "ckpt":             args.ckpt,
            "ckpt_basename":    ckpt_basename,
            "ckpt_sha1_8":      ckpt_sha1_8,
            "config":           args.config,
            "input_glob":       args.input,
            "resize_h":         int(H),
            "resize_w":         int(W),
            "num_classes":      int(args.num_classes),
            "seed":             int(args.seed),
            "deterministic":    bool(want_determinism),
            "device":           str(device),
            "cudnn_benchmark":  bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
    )
    print("[ARTIFACTS]", art.root)

    # Build model
    backbone   = "vit_large_patch14_reg4_dinov2" if "large" in os.path.basename(args.config).lower() \
                 else "vit_base_patch14_reg4_dinov2"
    num_q      = 100
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

    # Gather image paths
    paths = sorted(glob.glob(os.path.expanduser(args.input)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found for glob: {args.input}")

    anomaly_list: List[np.ndarray] = []
    ood_list:     List[np.ndarray] = []
    names:        List[str]        = []

    mask_logits_cache:  List[np.ndarray] = []
    class_logits_cache: List[np.ndarray] = []

    logits_h = logits_w = None
    unique_before_all: set = set()
    unique_after_all:  set = set()

    for p in paths:
        # Load and remap GT mask
        try:
            path_gt = gt_path_from_image(p)
            raw = np.array(Resize(size_hw, Image.NEAREST)(Image.open(path_gt)))
            unique_before_all.update(int(x) for x in np.unique(raw).tolist())

            ood = load_ood_mask(p, size_hw=size_hw)
            unique_after_all.update(int(x) for x in np.unique(ood).tolist())
        except Exception as e:
            print(f"[SKIP] GT error {p}: {e}")
            continue

        # Skip images with no OOD pixels (would distort metrics)
        if 1 not in np.unique(ood):
            continue

        # Forward pass
        img = Image.open(p).convert("RGB")
        x   = input_transform(img).unsqueeze(0).float().to(device)

        mask_logits, class_logits = model.forward_masks_and_classes(x)

        if logits_h is None:
            logits_h, logits_w = int(mask_logits.shape[-2]), int(mask_logits.shape[-1])

        # Compute anomaly score using the selected method
        if args.method == "rba":
            anomaly = rba_from_masks(mask_logits, class_logits, args.num_classes, args.temperature)
        elif args.method == "maxlogit":
            anomaly = anomaly_maxlogit_from_masks(mask_logits, class_logits, args.num_classes)
        else:
            # msp / maxentropy path
            pixel_probs = pixel_probs_from_masks(mask_logits, class_logits, args.num_classes, args.temperature)
            anomaly     = anomaly_from_pixel_probs(pixel_probs, args.method)

        # Upsample anomaly map back to input resolution if needed
        if anomaly.shape[-2:] != size_hw:
            anomaly = F.interpolate(
                anomaly.unsqueeze(1),
                size=size_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        anomaly_list.append(anomaly.squeeze(0).detach().cpu().float().numpy())
        ood_list.append(ood)
        names.append(os.path.basename(p))

        if args.save_logits:
            mask_logits_cache.append(mask_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())
            class_logits_cache.append(class_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())

    n_used = len(anomaly_list)
    if n_used == 0:
        raise RuntimeError(
            "No valid images used (all skipped or no OOD pixels after remapping). "
            f"mask_unique_before={sorted(unique_before_all)} "
            f"mask_unique_after={sorted(unique_after_all)}"
        )

    # Compute metrics
    ood_gts        = np.array(ood_list)
    anomaly_scores = np.array(anomaly_list)

    ood_out = anomaly_scores[ood_gts == 1]
    in_out  = anomaly_scores[ood_gts == 0]

    val_out   = np.concatenate([in_out, ood_out])
    val_label = np.concatenate([np.zeros(len(in_out)), np.ones(len(ood_out))])

    auprc = float(average_precision_score(val_label, val_out))
    fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

    metrics: Dict[str, Any] = {
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model":               "EoMT",
        "dataset":             args.dataset_name,
        "method":              args.method,
        "temperature":         float(args.temperature),
        "mode":                args.mode,
        "seed":                int(args.seed),
        "deterministic":       bool(want_determinism),
        "num_classes":         int(args.num_classes),
        "resize_h":            int(H),
        "resize_w":            int(W),
        "gt_h":                int(ood_gts.shape[-2]),
        "gt_w":                int(ood_gts.shape[-1]),
        "logits_h":            int(logits_h) if logits_h is not None else None,
        "logits_w":            int(logits_w) if logits_w is not None else None,
        "ckpt":                args.ckpt,
        "ckpt_basename":       ckpt_basename,
        "ckpt_sha1_8":         ckpt_sha1_8,
        "config":              args.config,
        "auprc":               auprc,
        "fpr95":               fpr95,
        "auprc_pct":           auprc * 100.0,
        "fpr95_pct":           fpr95 * 100.0,
        "images_used":         int(n_used),
        "device":              str(device),
        "cudnn_benchmark":     bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "mask_unique_before":  sorted(unique_before_all),
        "mask_unique_after":   sorted(unique_after_all),
    }

    print("=====================================")
    print(f"EoMT | dataset={args.dataset_name} | method={args.method} | T={args.temperature} | mode={args.mode}")
    print(f"AUPRC: {metrics['auprc_pct']:.4f}")
    print(f"FPR@95TPR: {metrics['fpr95_pct']:.4f}")
    print(f"Images used: {metrics['images_used']}")
    print(f"Resize: {H}x{W} | logits: {metrics['logits_h']}x{metrics['logits_w']}")
    print("=====================================")

    # Save results
    json_path = art.results / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {json_path}")

    csv_path = art.results / "metrics.csv"
    append_metrics_csv(csv_path, metrics)
    print(f"[SAVED] {csv_path}")

    # Optionally cache logits for offline temperature sweep
    if args.save_logits:
        ds = args.dataset_name
        np.save(art.logits / f"{ds}__mask_logits_f16.npy",  np.array(mask_logits_cache,  dtype=np.float16))
        np.save(art.logits / f"{ds}__class_logits_f16.npy", np.array(class_logits_cache, dtype=np.float16))
        np.save(art.logits / f"{ds}__gt.npy",               ood_gts.astype(np.uint8))
        with open(art.logits / f"{ds}__names.json", "w", encoding="utf-8") as f:
            json.dump(names, f, indent=2)

        print(f"[CACHED] {art.logits / f'{ds}__mask_logits_f16.npy'}")
        print(f"[CACHED] {art.logits / f'{ds}__class_logits_f16.npy'}")
        print(f"[CACHED] {art.logits / f'{ds}__gt.npy'}")
        print(f"[CACHED] {art.logits / f'{ds}__names.json'}")


if __name__ == "__main__":
    main()