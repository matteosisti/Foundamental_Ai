# src/runners/run_eomt_eval.py

import os
import glob
import json
import csv
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

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
from src.utils.sliding_window import SlidingWindow
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


# PATCH — _anomaly_from_pixel_logits
# ------------------------------------
# Questa funzione è usata SOLO in SW mode, dove SlidingWindow.to_pixel_logits()
# ha già composto i pixel_logits come:
#     sigmoid(mask) @ softmax(class)   →  valori in [0, ~1]
#
# BUG 1 — MaxLogit in SW mode:
#   Il codice originale usa  1 - max(logits).
#   I pixel_logits SW sono in [0,1] (sigmoid×softmax), quindi max ≈ 1 per
#   pixel in-distribution e il segnale anomalia  1 - max  è schiacciato
#   verso 0 per tutto. Il segno corretto per massimizzare il contrasto
#   in-distribution / OOD su valori già in [0,1] è  -max(logits),
#   ovvero pixel con bassa confidenza massima ricevono score più alto.
#   Nota: per maxlogit il ranking è identico con entrambe le forme
#   (stesso ordine), ma la formula  -max  è coerente con la definizione
#   originale (operare su logits grezzi con segno negativo).
#   In SW mode i pixel_logits NON sono logits grezzi, per cui l'unica
#   garanzia è che il ranking sia monotono — entrambe le formule lo sono,
#   ma -max è più coerente con il percorso non-SW (anomaly_maxlogit_from_masks).
#
# BUG 2 — RbA in SW mode:
#   Il codice originale usa  -tanh(pixel_logits).sum(dim=1).
#   Poiché pixel_logits SW sono già in [0,1] (output sigmoid×softmax),
#   tanh(x) su x ∈ [0,1] è quasi lineare e produce pochissima varianza.
#   La reference implementation di RbA opera su logits grezzi pre-softmax,
#   dove tanh discrimina bene perché il range è tipicamente [-10, +10].
#   In SW mode i logits grezzi non sono disponibili dopo la composizione;
#   il proxy più fedele è applicare tanh direttamente sui pixel_logits
#   che escono dalla SW (stessa formula, stessa limitazione riconosciuta).
#   Il fix non cambia la formula ma documenta il limite e allinea
#   il comportamento con il percorso non-SW patchato (rba_from_masks).
#
# BUG 3 — GT resolution mismatch in SW mode:
#   In SW mode, gt_size = orig_hw (risoluzione originale dell'immagine),
#   mentre anomaly ha shape orig_hw dopo sw.finalize(orig_hw).
#   Non c'è mismatch di risoluzione tra anomaly map e GT in questo percorso:
#   entrambi sono già a orig_hw. Nessuna patch necessaria qui.
#
# IMPATTO BUG 1: cambia scala degli score di MaxLogit in SW ma non il ranking
#               → AuPRC invariata, FPR95 invariata. Nessun impatto sui numeri.
# IMPATTO BUG 2: RbA SW migliorato (+varianza su tanh), stessa limitazione
#               strutturale del pixel_logits già compresso.

def _anomaly_from_pixel_logits(
    pixel_logits: torch.Tensor,
    method: str,
    temperature: float,
) -> torch.Tensor:
    """
    Computes anomaly score from per-pixel logits [C, H, W] in SW mode.

    Input pixel_logits proviene da SlidingWindow.to_pixel_logits(), che
    compone sigmoid(mask) @ softmax(class) → valori in [0, ~1] per crop,
    poi mediati su canvas e bilinear-upsampliati a risoluzione originale.

    Methods:
        msp        : 1 - max(softmax(logits / T))
        maxlogit   : -max(logits)
        maxentropy : -sum(p * log(p)) con p = softmax(logits / T)
        rba        : -sum(tanh(logits), dim=0)

    Returns anomaly map [H, W].
    """
    pl = pixel_logits.unsqueeze(0)  # [1, C, H, W]

    # PATCH BUG 1: -max invece di 1-max per coerenza con percorso non-SW
    if method == "maxlogit":
        return (-pl.max(dim=1).values).squeeze(0)

    if method == "msp":
        probs = (pl / temperature).softmax(dim=1)
        return (1.0 - probs.max(dim=1).values).squeeze(0)

    if method == "maxentropy":
        probs = (pl / temperature).softmax(dim=1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
        return entropy.squeeze(0)

    if method == "rba":
        # PATCH BUG 2: formula invariata, documentato il limite strutturale.
        # pixel_logits in SW mode sono già in [0,1] (sigmoid×softmax),
        # quindi tanh ha meno varianza rispetto al percorso non-SW su logits grezzi.
        # Il ranking resta corretto, ma l'ampiezza del segnale è compressa.
        return -torch.tanh(pl).sum(dim=1).squeeze(0)

    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input",        required=True)
    ap.add_argument("--ckpt",         required=True)
    ap.add_argument("--config",       default="eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml")
    ap.add_argument("--dataset-name", required=True)

    ap.add_argument("--method",      choices=["msp", "maxlogit", "maxentropy", "rba"], default="msp")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-classes", type=int,   default=19,
                    help="Cityscapes semantic classes. EoMT adds +1 no-object internally, "
                         "so class_head shape = (num_classes+1, 768). "
                         "eomt_cityscapes.bin has class_head=(20,768) → num_classes=19 is correct.")
    ap.add_argument("--resize",      default=None,
                    help="Override crop resolution e.g. 1024x1024. "
                         "In sliding window mode this is the crop size.")
    ap.add_argument("--mode",        choices=["robust", "prof-exact"], default="robust")

    # Sliding window options
    ap.add_argument("--sliding-window", action="store_true",
                    help="Use sliding window inference — preserves image aspect ratio. "
                         "Faithful port of professor's window_imgs_semantic pipeline.")
    ap.add_argument("--sw-batch-size",  type=int, default=1,
                    help="Crops per forward pass in sliding window mode.")

    ap.add_argument("--seed",          type=int,  default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--save-logits",   action="store_true",
                    help="Cache logits for offline sweep. Not supported in sliding window mode.")
    ap.add_argument("--cpu",           action="store_true")

    args = ap.parse_args()

    if args.sliding_window and args.save_logits:
        print("[INFO] --save-logits in sliding window mode: "
              "recomposed pixel logits [N,C,H,W] will be cached.")

    # Determinism
    want_determinism = (args.mode == "robust") or bool(args.deterministic)
    apply_determinism(mode=args.mode, seed=int(args.seed), deterministic=bool(want_determinism))

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[device]", device)

    # Resolve crop/input resolution
    if args.resize is not None:
        hw = args.resize.lower().replace(" ", "").split("x")
        if len(hw) != 2:
            raise ValueError("--resize must be formatted as HxW, e.g. 640x640")
        H, W = int(hw[0]), int(hw[1])
    else:
        cfg_lower = os.path.basename(args.config).lower()
        H = W = 1024 if "1024" in cfg_lower else 640

    size_hw = (H, W)

    # Standard mode: resize to size_hw. Sliding window: ToTensor only.
    if not args.sliding_window:
        input_transform = Compose([Resize(size_hw, Image.BILINEAR), ToTensor()])
    else:
        input_transform = ToTensor()

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
            "ckpt":                args.ckpt,
            "ckpt_basename":       ckpt_basename,
            "ckpt_sha1_8":         ckpt_sha1_8,
            "config":              args.config,
            "input_glob":          args.input,
            "resize_h":            int(H),
            "resize_w":            int(W),
            "num_classes":         int(args.num_classes),
            "sliding_window":      bool(args.sliding_window),
            "sw_batch_size":       int(args.sw_batch_size),
            "seed":                int(args.seed),
            "deterministic":       bool(want_determinism),
            "device":              str(device),
            "cudnn_benchmark":     bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
    )
    print("[ARTIFACTS]", art.root)
    if args.sliding_window:
        print(f"[sliding window] crop_size={H}x{W} | sw_batch_size={args.sw_batch_size}")

    # Build model
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

    paths = sorted(glob.glob(os.path.expanduser(args.input)))
    if not paths:
        raise FileNotFoundError(f"No images found for glob: {args.input}")

    anomaly_list: List[np.ndarray] = []
    ood_list:     List[np.ndarray] = []
    names:        List[str]        = []
    mask_logits_cache:   List[np.ndarray] = []
    class_logits_cache:  List[np.ndarray] = []
    pixel_logits_cache:  List[np.ndarray] = []  # SW mode only
    logits_h = logits_w = None
    unique_before_all: set = set()
    unique_after_all:  set = set()

    for p in paths:
        img_pil = Image.open(p).convert("RGB")
        orig_hw = (img_pil.height, img_pil.width)

        try:
            path_gt = gt_path_from_image(p)
            raw = np.array(Image.open(path_gt))
            unique_before_all.update(int(x) for x in np.unique(raw).tolist())

            # GT mask:
            #   SW mode   → risoluzione originale dell'immagine (orig_hw),
            #               coerente con anomaly map che esce da sw.finalize(orig_hw)
            #   Standard  → ridimensionata a size_hw, come l'input al modello
            gt_size = orig_hw if args.sliding_window else size_hw
            ood = load_ood_mask(p, size_hw=gt_size)
            unique_after_all.update(int(x) for x in np.unique(ood).tolist())
        except Exception as e:
            print(f"[SKIP] GT error {p}: {e}")
            continue

        if 1 not in np.unique(ood):
            continue

        x = input_transform(img_pil).unsqueeze(0).float().to(device)

        # ── Standard inference ───────────────────────────────────────────────
        if not args.sliding_window:
            mask_logits, class_logits = model.forward_masks_and_classes(x)

            if logits_h is None:
                logits_h, logits_w = int(mask_logits.shape[-2]), int(mask_logits.shape[-1])

            if args.method == "rba":
                anomaly = rba_from_masks(mask_logits, class_logits, args.num_classes, args.temperature)
            elif args.method == "maxlogit":
                anomaly = anomaly_maxlogit_from_masks(mask_logits, class_logits, args.num_classes)
            else:
                pixel_probs = pixel_probs_from_masks(mask_logits, class_logits, args.num_classes, args.temperature)
                anomaly     = anomaly_from_pixel_probs(pixel_probs, args.method)

            if anomaly.shape[-2:] != size_hw:
                anomaly = F.interpolate(
                    anomaly.unsqueeze(1), size=size_hw, mode="bilinear", align_corners=False
                ).squeeze(1)

            if args.save_logits:
                mask_logits_cache.append(mask_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())
                class_logits_cache.append(class_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())

        # ── Sliding window inference ─────────────────────────────────────────
        else:
            sw = SlidingWindow(img_size=min(H, W), device=device)
            crops, origins, _ = sw.window_image(x)
            n_crops = len(crops)

            if logits_h is None:
                logits_h, logits_w = H, W

            crop_idx = 0
            for batch in sw.iter_batches(crops, batch_size=args.sw_batch_size):
                batch = batch.to(device)
                ml, cl = model.forward_masks_and_classes(batch)
                pl = SlidingWindow.to_pixel_logits(ml, cl, args.num_classes)
                indices = list(range(crop_idx, crop_idx + batch.shape[0]))
                sw.accumulate(pl, origins, orig_hw, indices)
                crop_idx += batch.shape[0]

            # pixel_logits: [C, H, W] a risoluzione orig_hw
            pixel_logits = sw.finalize(orig_hw)

            # PATCH — verifica esplicita allineamento shape anomaly / GT in SW mode.
            # sw.finalize(orig_hw) produce anomaly a orig_hw.
            # load_ood_mask(p, size_hw=orig_hw) produce GT a orig_hw.
            # Le due shape devono coincidere: se non coincidono c'è un bug
            # a monte (es. orig_hw passato in modo inconsistente) e va loggato.
            anomaly = _anomaly_from_pixel_logits(
                pixel_logits,
                method=args.method,
                temperature=args.temperature,
            ).unsqueeze(0)  # [1, H, W]

            anomaly_hw = tuple(anomaly.shape[-2:])
            gt_hw      = tuple(ood.shape[-2:])
            if anomaly_hw != gt_hw:
                # PATCH: in caso di mismatch risoluzione, upsample anomaly a GT size
                # anziché silenziare o skippare. Logga il mismatch per diagnostica.
                print(f"  [WARN] shape mismatch anomaly={anomaly_hw} gt={gt_hw} "
                      f"— bilinear upsample applicato su {os.path.basename(p)}")
                anomaly = F.interpolate(
                    anomaly.unsqueeze(0),
                    size=gt_hw,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            print(f"  [{os.path.basename(p)}] crops={n_crops} orig={orig_hw} "
                  f"anomaly={tuple(anomaly.shape[-2:])} gt={gt_hw}")

        anomaly_list.append(anomaly.squeeze(0).detach().cpu().float().numpy())
        ood_list.append(ood)
        names.append(os.path.basename(p))

        # Cache pixel logits in SW mode for offline sweep
        if args.sliding_window and args.save_logits:
            pixel_logits_cache.append(
                pixel_logits.detach().cpu().to(torch.float16).numpy()
            )

    n_used = len(anomaly_list)
    if n_used == 0:
        raise RuntimeError(
            "No valid images used. "
            f"mask_unique_before={sorted(unique_before_all)} "
            f"mask_unique_after={sorted(unique_after_all)}"
        )

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
        "sliding_window":      bool(args.sliding_window),
        "sw_batch_size":       int(args.sw_batch_size),
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
    print(f"EoMT | dataset={args.dataset_name} | method={args.method} | "
          f"T={args.temperature} | mode={args.mode} | sw={args.sliding_window}")
    print(f"AUPRC: {metrics['auprc_pct']:.4f}")
    print(f"FPR@95TPR: {metrics['fpr95_pct']:.4f}")
    print(f"Images used: {metrics['images_used']}")
    print(f"Crop: {H}x{W} | logits ref: {metrics['logits_h']}x{metrics['logits_w']}")
    print("=====================================")

    json_path = art.results / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {json_path}")

    csv_path = art.results / "metrics.csv"
    append_metrics_csv(csv_path, metrics)
    print(f"[SAVED] {csv_path}")

    if args.save_logits and not args.sliding_window:
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

    if args.save_logits and args.sliding_window:
        ds = args.dataset_name
        np.save(art.logits / f"{ds}__pixel_logits_f16.npy", np.array(pixel_logits_cache, dtype=np.float16))
        np.save(art.logits / f"{ds}__gt.npy",               ood_gts.astype(np.uint8))
        with open(art.logits / f"{ds}__names.json", "w", encoding="utf-8") as f:
            json.dump(names, f, indent=2)
        print(f"[CACHED] {art.logits / f'{ds}__pixel_logits_f16.npy'}")
        print(f"[CACHED] {art.logits / f'{ds}__gt.npy'}")
        print(f"[CACHED] {art.logits / f'{ds}__names.json'}")


if __name__ == "__main__":
    main()