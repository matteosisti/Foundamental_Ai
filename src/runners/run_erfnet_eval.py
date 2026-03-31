# src/runners/run_erfnet_eval.py
#
# Unified ERFNet OOD evaluation runner (robust + prof-exact) with:
# - deterministic control via src.utils.determinism.apply_determinism
# - artifacts folder auto-run dir + config.json via src.utils.artifacts.create_run_dir
# - optional caching of RAW logits for temperature sweeps

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
from src.utils.determinism import apply_determinism
from src.utils.ood_dataset import gt_path_from_image, load_ood_mask


NUM_CLASSES = 20


# ---------------------------------------------------------------------------
# ERFNet-specific helpers (not in ood_dataset — model loading and scoring)
# ---------------------------------------------------------------------------

def _weights_meta(weights_path: str) -> dict:
    p = Path(os.path.expanduser(weights_path))
    meta = {
        "weights":          str(p),
        "weights_basename": p.name,
    }
    try:
        st = p.stat()
        meta["weights_size_bytes"] = int(st.st_size)
        meta["weights_mtime"]      = float(st.st_mtime)
    except Exception:
        pass
    return meta


def load_erfnet(weights_path: str, device: torch.device, mode: str) -> torch.nn.Module:
    model = ERFNet(NUM_CLASSES).to(device)

    state = torch.load(os.path.expanduser(weights_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    own = model.state_dict()

    if mode == "prof-exact":
        loadable = {}
        for k, v in state.items():
            k2 = k.replace("module.", "")
            if k2 in own and own[k2].shape == v.shape:
                loadable[k2] = v
        model.load_state_dict(loadable, strict=False)
    else:
        for k, v in state.items():
            k2 = k.replace("module.", "")
            if k2 in own and own[k2].shape == v.shape:
                own[k2].copy_(v)
        model.load_state_dict(own, strict=False)

    model.eval()
    return model


def anomaly_from_logits(logits: torch.Tensor, method: str, T: float) -> np.ndarray:
    """
    Compute per-pixel anomaly score from ERFNet logits [1, C, H, W].

    Methods:
        maxlogit   — -max(logits)          [temperature has no effect]
        msp        — 1 - max(softmax(L/T))
        maxentropy — entropy(softmax(L/T))
    """
    if method == "maxlogit":
        m = logits.max(dim=1).values
        return (-m).squeeze(0).detach().cpu().float().numpy()

    p = F.softmax(logits / T, dim=1)

    if method == "msp":
        return (1.0 - p.max(dim=1).values).squeeze(0).detach().cpu().float().numpy()

    if method == "maxentropy":
        ent = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input",        required=True, help="Glob pattern, e.g. /path/images/*.*")
    ap.add_argument("--weights",      required=True, help="Path to ERFNet .pth checkpoint")
    ap.add_argument("--dataset-name", required=True, help="Short name e.g. RA21")
    ap.add_argument("--method",       choices=["msp", "maxlogit", "maxentropy"], default="msp")
    ap.add_argument("--temperature",  type=float, default=1.0)
    ap.add_argument("--mode",         choices=["robust", "prof-exact"], default="robust")

    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--seed",          type=int,  default=0)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--artifacts-dir",     default="artifacts")
    ap.add_argument("--save-logits",       action="store_true")
    ap.add_argument("--save-anomaly-maps", action="store_true")

    args = ap.parse_args()

    # Determinism policy
    want_determinism = (args.mode == "robust") or bool(args.deterministic)
    apply_determinism(mode=args.mode, seed=int(args.seed), deterministic=bool(want_determinism))

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    resize_h, resize_w = 512, 1024
    size_hw = (resize_h, resize_w)

    input_transform = Compose([
        Resize(size_hw, Image.BILINEAR),
        ToTensor(),
    ])

    extra = {
        "input_glob":          args.input,
        "resize_h":            int(resize_h),
        "resize_w":            int(resize_w),
        "num_classes":         int(NUM_CLASSES),
        "device":              str(device),
        "seed":                int(args.seed),
        "deterministic":       bool(want_determinism),
        "cudnn_benchmark":     bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    extra.update(_weights_meta(args.weights))

    art = create_run_dir(
        artifacts_root=args.artifacts_dir,
        dataset=args.dataset_name,
        model="ERFNet",
        method=args.method,
        temperature=args.temperature,
        mode=args.mode,
        extra=extra,
    )
    print("[ARTIFACTS]", art.root)

    model = load_erfnet(args.weights, device=device, mode=args.mode)

    paths = sorted(glob.glob(os.path.expanduser(args.input)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found: {args.input}")

    anomaly_list = []
    ood_list     = []
    logits_cache = []
    gt_cache     = []
    names_cache  = []

    unique_before_all: set = set()
    unique_after_all:  set = set()

    for path in paths:
        try:
            path_gt = gt_path_from_image(path)
            raw = np.array(Resize(size_hw, Image.NEAREST)(Image.open(path_gt)))
            unique_before_all.update(int(x) for x in np.unique(raw).tolist())

            ood = load_ood_mask(path, size_hw=size_hw)
            unique_after_all.update(int(x) for x in np.unique(ood).tolist())
        except Exception as e:
            print(f"[SKIP] GT error {path}: {e}")
            continue

        if 1 not in np.unique(ood):
            continue

        img    = Image.open(path).convert("RGB")
        x      = input_transform(img).unsqueeze(0).float().to(device)
        logits = model(x)

        if args.save_logits:
            logits_cache.append(logits.squeeze(0).detach().cpu().numpy().astype(np.float32))
            gt_cache.append(ood.astype(np.uint8))
            names_cache.append(os.path.basename(path))

        anomaly = anomaly_from_logits(logits, args.method, args.temperature)
        ood_list.append(ood)
        anomaly_list.append(anomaly)

        if args.save_anomaly_maps:
            np.save(art.anomaly_maps / f"{os.path.basename(path)}.npy", anomaly.astype(np.float32))

    if len(ood_list) == 0:
        raise RuntimeError(
            "No valid images used (all skipped or no OOD pixels). "
            f"mask_unique_before={sorted(unique_before_all)} "
            f"mask_unique_after={sorted(unique_after_all)}"
        )

    ood_gts        = np.array(ood_list)
    anomaly_scores = np.array(anomaly_list)

    ood_out = anomaly_scores[ood_gts == 1]
    ind_out = anomaly_scores[ood_gts == 0]

    val_out   = np.concatenate([ind_out, ood_out])
    val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])

    auprc = float(average_precision_score(val_label, val_out))
    fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

    metrics = {
        "model":               "ERFNet",
        "dataset":             args.dataset_name,
        "method":              args.method,
        "temperature":         float(args.temperature),
        "mode":                args.mode,
        "seed":                int(args.seed),
        "deterministic":       bool(want_determinism),
        "auprc":               auprc,
        "fpr95":               fpr95,
        "auprc_pct":           auprc * 100.0,
        "fpr95_pct":           fpr95 * 100.0,
        "images_used":         int(len(ood_list)),
        "input_glob":          args.input,
        "weights":             os.path.expanduser(args.weights),
        "device":              str(device),
        "resize_h":            int(resize_h),
        "resize_w":            int(resize_w),
        "gt_h":                int(ood_gts.shape[-2]),
        "gt_w":                int(ood_gts.shape[-1]),
        "num_classes":         int(NUM_CLASSES),
        "cudnn_benchmark":     bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "mask_unique_before":  sorted(unique_before_all),
        "mask_unique_after":   sorted(unique_after_all),
    }

    print("=====================================")
    print(f"ERFNet | dataset={args.dataset_name} | method={args.method} | T={args.temperature} | mode={args.mode}")
    print(f"AUPRC: {metrics['auprc_pct']:.4f}")
    print(f"FPR@95TPR: {metrics['fpr95_pct']:.4f}")
    print(f"Images used: {metrics['images_used']}")
    print("=====================================")

    json_path = art.results / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {json_path}")

    csv_path = art.results / "metrics.csv"
    append_metrics_csv(csv_path, metrics)
    print(f"[SAVED] {csv_path}")

    if args.save_logits and len(logits_cache) > 0:
        ds = args.dataset_name
        np.save(art.logits / f"{ds}__logits.npy", np.stack(logits_cache, axis=0))
        np.save(art.logits / f"{ds}__gt.npy",     np.stack(gt_cache,     axis=0))
        with open(art.logits / f"{ds}__names.json", "w", encoding="utf-8") as f:
            json.dump(names_cache, f, indent=2)

        print(f"[CACHED] {art.logits / f'{ds}__logits.npy'}")
        print(f"[CACHED] {art.logits / f'{ds}__gt.npy'}")
        print(f"[CACHED] {art.logits / f'{ds}__names.json'}")


if __name__ == "__main__":
    main()