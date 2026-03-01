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

# Assuming ERFNet is in your python path (e.g., eval/erfnet.py)
from eval.erfnet import ERFNet

from src.utils.artifacts import create_run_dir
from src.utils.ood_metrics import fpr_at_95_tpr
from src.utils.determinism import set_determinism

NUM_CLASSES = 20

def load_erfnet(weights_path: str, device: torch.device, mode: str) -> torch.nn.Module:
    """
    Load ERFNet weights with flexible mapping for 'module.' prefixes.
    'prof-exact' follows a stricter loading policy.
    """
    model = ERFNet(NUM_CLASSES).to(device)
    state = torch.load(weights_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    own = model.state_dict()
    loadable = {}

    for k, v in state.items():
        k_clean = k.replace("module.", "")
        if k_clean in own and own[k_clean].shape == v.shape:
            if mode == "prof-exact":
                loadable[k_clean] = v
            else:
                own[k_clean].copy_(v)

    if mode == "prof-exact":
        model.load_state_dict(loadable, strict=False)
    else:
        model.load_state_dict(own, strict=False)

    model.eval()
    return model

def gt_path_from_image(path_img: str) -> str:
    """Derive Ground Truth path from image path based on dataset structure."""
    path_gt = path_img.replace("images", "labels_masks")
    if any(x in path_gt for x in ["RoadObstacle21", "fs_static", "RoadAnomaly", "LostAndFound"]):
        return os.path.splitext(path_gt)[0] + ".png"
    return path_gt if os.path.splitext(path_gt)[1] else (path_gt + ".png")

def remap_ood_mask(path_gt: str, ood: np.ndarray) -> np.ndarray:
    """Standardize OOD labels: 1 for Anomaly, 0 for In-Distribution, 255 for Ignore."""
    if "RoadAnomaly" in path_gt:
        ood = np.where((ood == 2), 1, ood)
    elif "LostAndFound" in path_gt or "FS_LostFound_full" in path_gt:
        ood = np.where((ood == 0), 255, ood)
        ood = np.where((ood == 1), 0, ood)
        ood = np.where((ood > 1) & (ood < 201), 1, ood)
    elif "Streethazard" in path_gt:
        ood = np.where((ood == 14), 255, ood)
        ood = np.where((ood < 20), 0, ood)
        ood = np.where((ood == 255), 1, ood)
    return ood

def load_ood_mask(path_img: str, target_transform: Compose) -> np.ndarray:
    path_gt = gt_path_from_image(path_img)
    mask = Image.open(path_gt)
    mask = target_transform(mask)
    ood = np.array(mask)
    return remap_ood_mask(path_gt, ood)

def anomaly_from_logits(logits: torch.Tensor, method: str, T: float) -> np.ndarray:
    """Compute anomaly score map from raw logits."""
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
    ap.add_argument("--input", required=True, help="Glob pattern for images")
    ap.add_argument("--weights", required=True, help="Path to .pth ERFNet weights")
    ap.add_argument("--dataset-name", required=True, help="Short name (e.g. RA21)")
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

    # Determinism logic
    want_determinism = True if args.mode == "robust" else bool(args.deterministic)
    set_determinism(seed=args.seed, deterministic=want_determinism)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Image transforms
    resize_h, resize_w = 512, 1024
    input_transform = Compose([Resize((resize_h, resize_w), Image.BILINEAR), ToTensor()])
    target_transform = Compose([Resize((resize_h, resize_w), Image.NEAREST)])

    # Setup Artifacts
    art = create_run_dir(
        artifacts_root=args.artifacts_dir,
        dataset=args.dataset_name,
        model="ERFNet",
        method=args.method,
        temperature=args.temperature,
        mode=args.mode,
        extra={"weights": args.weights, "seed": args.seed, "deterministic": want_determinism}
    )

    model = load_erfnet(args.weights, device=device, mode=args.mode)
    paths = sorted(glob.glob(os.path.expanduser(args.input)))
    
    if not paths:
        raise FileNotFoundError(f"No images found: {args.input}")

    anomaly_list, ood_list = [], []
    logits_cache, gt_cache, names_cache = [], [], []

    print(f"Starting evaluation on {len(paths)} images...")

    for path in paths:
        try:
            ood = load_ood_mask(path, target_transform=target_transform)
            if 1 not in np.unique(ood): continue # Skip if no anomalies present

            img = Image.open(path).convert("RGB")
            x = input_transform(img).unsqueeze(0).to(device)
            
            logits = model(x) # [1, C, H, W]

            if args.save_logits:
                logits_cache.append(logits.squeeze(0).cpu().numpy().astype(np.float32))
                gt_cache.append(ood.astype(np.uint8))
                names_cache.append(os.path.basename(path))

            anomaly = anomaly_from_logits(logits, args.method, args.temperature)
            ood_list.append(ood)
            anomaly_list.append(anomaly)

            if args.save_anomaly_maps:
                np.save(art.anomaly_maps / f"{os.path.basename(path)}.npy", anomaly)

        except Exception as e:
            print(f"[SKIP] Error processing {path}: {e}")

    if not ood_list:
        raise RuntimeError("No valid images with OOD pixels found.")

    # Metric Calculation
    ood_gts = np.array(ood_list)
    anomaly_scores = np.array(anomaly_list)
    
    ood_mask = (ood_gts == 1)
    in_mask = (ood_gts == 0)
    
    val_out = np.concatenate([anomaly_scores[in_mask], anomaly_scores[ood_mask]])
    val_label = np.concatenate([np.zeros(in_mask.sum()), np.ones(ood_mask.sum())])

    metrics = {
        "auprc_pct": float(average_precision_score(val_label, val_out)) * 100.0,
        "fpr95_pct": float(fpr_at_95_tpr(val_out, val_label, mode=args.mode)) * 100.0,
        "images_used": len(ood_list)
    }

    print(f"\nResults: AUPRC={metrics['auprc_pct']:.2f}% | FPR95={metrics['fpr95_pct']:.2f}%")

    # Save outputs
    with open(art.results / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    append_metrics_csv(art.results / "metrics.csv", metrics)

    if args.save_logits and logits_cache:
        np.save(art.logits / f"{args.dataset_name}__logits.npy", np.stack(logits_cache))
        np.save(art.logits / f"{args.dataset_name}__gt.npy", np.stack(gt_cache))
        print(f"Logits cached for temperature sweep in {art.logits}")

if __name__ == "__main__":
    main()