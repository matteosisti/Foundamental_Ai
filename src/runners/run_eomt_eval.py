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


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _dbg_tensor(tag: str, t: torch.Tensor) -> None:
    """Stampa shape, dtype, device, min/mean/max/nan/inf di un tensore."""
    if t.numel() == 0:
        print(f"  [DBG] {tag}: EMPTY tensor shape={tuple(t.shape)}")
        return
    tf = t.float()
    print(
        f"  [DBG] {tag}: shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} "
        f"min={tf.min().item():.4f} mean={tf.mean().item():.4f} "
        f"max={tf.max().item():.4f} "
        f"nan={torch.isnan(tf).sum().item()} "
        f"inf={torch.isinf(tf).sum().item()}"
    )


def _dbg_array(tag: str, a: np.ndarray) -> None:
    """Stampa shape, dtype, unique values di un array numpy (tipicamente GT mask)."""
    print(
        f"  [DBG] {tag}: shape={a.shape} dtype={a.dtype} "
        f"unique={np.unique(a).tolist()}"
    )


def _dbg_anomaly_score_distribution(tag: str, scores: np.ndarray, gt: np.ndarray) -> None:
    """
    Stampa la distribuzione degli anomaly score separata per InD (0) e OOD (1),
    più quanti pixel sono stati ignorati (255).
    Utile per capire se gli score sono invertiti o collassati.
    """
    ind_scores = scores[gt == 0]
    ood_scores = scores[gt == 1]
    ign_count  = int((gt == 255).sum())

    def _stats(arr: np.ndarray, name: str) -> str:
        if len(arr) == 0:
            return f"{name}: N=0"
        return (
            f"{name}: N={len(arr)} "
            f"min={arr.min():.4f} p25={np.percentile(arr,25):.4f} "
            f"med={np.median(arr):.4f} p75={np.percentile(arr,75):.4f} "
            f"max={arr.max():.4f}"
        )

    print(f"  [DBG] {tag} score distribution:")
    print(f"    {_stats(ind_scores, 'InD')}")
    print(f"    {_stats(ood_scores, 'OOD')}")
    print(f"    ignored pixels (255): {ign_count}")
    if len(ind_scores) > 0 and len(ood_scores) > 0:
        # Se med(OOD) < med(InD) gli score sono invertiti — anomalie hanno score basso
        direction = "OK (OOD > InD)" if np.median(ood_scores) > np.median(ind_scores) else "!! INVERTITO (OOD < InD) !!"
        print(f"    score direction: {direction}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Anomaly scoring — SW mode
# ---------------------------------------------------------------------------

# PATCH — _anomaly_from_pixel_logits
# In SW mode i pixel_logits escono da SlidingWindow.to_pixel_logits() come
# sigmoid(mask) @ softmax(class) → valori in [0, ~1].
# MaxLogit: -max (coerente con percorso non-SW)
# RbA: tanh su [0,1] → poca varianza, limite strutturale documentato

def _anomaly_from_pixel_logits(
    pixel_logits: torch.Tensor,
    method: str,
    temperature: float,
) -> torch.Tensor:
    """
    Computes anomaly score from per-pixel logits [C, H, W] in SW mode.

    Input pixel_logits: sigmoid(mask) @ softmax(class) → [0, ~1].

    Methods:
        msp        : 1 - max(softmax(logits / T))
        maxlogit   : -max(logits)
        maxentropy : -sum(p * log(p)), p = softmax(logits / T)
        rba        : -sum(tanh(logits), dim=0)

    Returns anomaly map [H, W].
    """
    pl = pixel_logits.unsqueeze(0)  # [1, C, H, W]

    # PATCH: -max invece di 1-max (stesso ranking, coerenza con percorso non-SW)
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
        # pixel_logits SW in [0,1]: tanh quasi lineare, varianza compressa.
        # Stessa formula della reference ma su input già compresso.
        return -torch.tanh(pl).sum(dim=1).squeeze(0)

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    ap.add_argument("--resize",      default=None)
    ap.add_argument("--mode",        choices=["robust", "prof-exact"], default="robust")

    ap.add_argument("--sliding-window", action="store_true")
    ap.add_argument("--sw-batch-size",  type=int, default=1)

    ap.add_argument("--seed",          type=int,  default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--save-logits",   action="store_true")
    ap.add_argument("--cpu",           action="store_true")

    # Flag debug: stampa info dettagliate per ogni immagine
    ap.add_argument("--debug", action="store_true",
                    help="Abilita print dettagliato per GT mask, logits, anomaly scores.")
    ap.add_argument("--no-totensor", action="store_true",
                    help="NON divide per 255 — passa uint8 [0,255] al modello come fa il gruppo 5. "
                         "Solo per confronto sperimentale.")
    ap.add_argument("--lightning-sw", action="store_true",
                    help="Usa window_imgs_semantic / revert_window_logits_semantic del Lightning "
                         "module invece del SlidingWindow custom. Replica fedelmente il percorso "
                         "del gruppo 5 (project.ipynb, semantic_inference). "
                         "Richiede che il modello sia caricato come MaskClassificationSemantic.")

    args = ap.parse_args()

    if args.sliding_window and args.save_logits:
        print("[INFO] --save-logits in SW mode: pixel logits [N,C,H,W] saranno cachati.")

    # --- Determinism ---
    want_determinism = (args.mode == "robust") or bool(args.deterministic)
    apply_determinism(mode=args.mode, seed=int(args.seed), deterministic=bool(want_determinism))

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    # --- Risoluzione ---
    if args.resize is not None:
        hw = args.resize.lower().replace(" ", "").split("x")
        if len(hw) != 2:
            raise ValueError("--resize deve essere HxW, es. 640x640")
        H, W = int(hw[0]), int(hw[1])
    else:
        cfg_lower = os.path.basename(args.config).lower()
        H = W = 1024 if "1024" in cfg_lower else 640

    size_hw = (H, W)

    # PATCH --no-totensor: replica il comportamento del gruppo 5 che passa
    # uint8 [0,255] direttamente al modello senza ToTensor() (divisione per 255).
    # Il modello EoMT fa internamente (x - mean) / std aspettandosi [0,1] float,
    # ma con [0,255] uint8 i logits escono con range molto più ampio.
    # Usare SOLO per confronto sperimentale — non è il comportamento corretto.
    if not args.sliding_window:
        if args.no_totensor:
            # uint8 [0,255] come gruppo 5: Resize PIL → np.array → permute → float (no /255)
            input_transform = Compose([Resize(size_hw, Image.BILINEAR)])
        else:
            input_transform = Compose([Resize(size_hw, Image.BILINEAR), ToTensor()])
    else:
        if args.no_totensor:
            input_transform = None  # gestito inline sotto
        else:
            input_transform = ToTensor()

    # --- Header di sessione ---
    print("=" * 60)
    print(f"[SESSION] dataset={args.dataset_name} method={args.method} T={args.temperature}")
    print(f"[SESSION] mode={args.mode} sw={args.sliding_window} sw_batch={args.sw_batch_size}")
    print(f"[SESSION] ckpt={args.ckpt}")
    print(f"[SESSION] config={args.config}")
    print(f"[SESSION] resize={H}x{W} num_classes={args.num_classes}")
    print(f"[SESSION] device={device} seed={args.seed} deterministic={want_determinism}")
    print(f"[SESSION] debug={args.debug}")
    print(f"[SESSION] no_totensor={args.no_totensor}  (uint8 [0,255] → modello)")
    print(f"[SESSION] lightning_sw={args.lightning_sw}  (usa window_imgs_semantic del Lightning module)")
    print("=" * 60)

    ckpt_basename = os.path.basename(args.ckpt)
    ckpt_sha1_8   = _sha1_8_of_file(args.ckpt)
    print(f"[CKPT] {ckpt_basename}  sha1_8={ckpt_sha1_8}")

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
    print(f"[ARTIFACTS] {art.root}")

    # --- Build model ---
    backbone = (
        "vit_large_patch14_reg4_dinov2"
        if "large" in os.path.basename(args.config).lower()
        else "vit_base_patch14_reg4_dinov2"
    )
    print(f"[MODEL] backbone={backbone}")

    # ── Model loading ────────────────────────────────────────────────────
    # Branch A (default):  EoMTWrapper custom — forward_masks_and_classes()
    # Branch B (--lightning-sw): MaskClassificationSemantic Lightning module
    #   usa window_imgs_semantic + to_per_pixel_logits_semantic +
    #   revert_window_logits_semantic esattamente come il gruppo 5.

    if args.lightning_sw:
        # ── BRANCH B: Lightning module (replica gruppo 5 / step8) ────────────
        # Istanzia MaskClassificationSemantic con l'img_size rilevata dal
        # checkpoint (1024×1024 per eomt_cityscapes.bin), carica i pesi
        # correttamente e usa window_imgs_semantic per l'inference.
        # Differenza chiave vs Branch A (EoMTWrapper a 640×640):
        #   - pos_embed caricato correttamente (no size mismatch)
        #   - window_imgs_semantic scala le immagini a 1024×1024 prima dei crop
        print("[MODEL] Caricamento via MaskClassificationSemantic (Lightning branch) ...")
        import importlib, yaml, sys as _sys

        _eomt_dir = os.path.join(os.getcwd(), "eomt")
        if _eomt_dir not in _sys.path:
            _sys.path.insert(0, _eomt_dir)
            print(f"[MODEL][lit] sys.path += {_eomt_dir}")

        try:
            import lightning as _lt
            print(f"[MODEL][lit] lightning {_lt.__version__}")
        except ImportError:
            import subprocess as _sp
            _sp.run([_sys.executable, "-m", "pip", "install", "-q", "lightning"], check=True)
            print("[MODEL][lit] lightning installato ✓")

        # Rileva img_size dal checkpoint ispezionando pos_embed
        _ckpt_raw = torch.load(args.ckpt, map_location="cpu")
        _ckpt_state = _ckpt_raw.get("state_dict", _ckpt_raw)
        _lit_img_size = (1024, 1024)  # default per eomt_cityscapes.bin
        for _pk, _pv in _ckpt_state.items():
            if "pos_embed" in _pk and _pv.dim() == 3:
                _seq = _pv.shape[1]
                if _seq in (4096, 4101):
                    _lit_img_size = (1024, 1024)
                elif _seq in (1600, 1605):
                    _lit_img_size = (640, 640)
                print(f"[MODEL][lit] pos_embed seq={_seq} → img_size={_lit_img_size}")
                break

        with open(args.config) as _f:
            _cfg = yaml.safe_load(_f)

        # Encoder a _lit_img_size
        _ec = _cfg["model"]["init_args"]["network"]["init_args"]["encoder"]
        _em, _en = _ec["class_path"].rsplit(".", 1)
        _encoder = getattr(importlib.import_module(_em), _en)(
            img_size=_lit_img_size, **_ec.get("init_args", {})
        )

        # Network
        _nc = _cfg["model"]["init_args"]["network"]
        _nm, _nn = _nc["class_path"].rsplit(".", 1)
        _nkw = {k: v for k, v in _nc["init_args"].items()
                if k not in ("encoder", "num_classes", "masked_attn_enabled")}
        _network = getattr(importlib.import_module(_nm), _nn)(
            masked_attn_enabled=False,
            num_classes=args.num_classes,
            encoder=_encoder,
            **_nkw,
        )

        # Lightning module
        _lm, _ln = _cfg["model"]["class_path"].rsplit(".", 1)
        _lkw = {k: v for k, v in _cfg["model"]["init_args"].items() if k != "network"}
        for _rk in ("stuff_classes", "overlap_thresh", "mask_thresh"):
            _lkw.pop(_rk, None)
        model = getattr(importlib.import_module(_lm), _ln)(
            network=_network,
            img_size=_lit_img_size,
            num_classes=args.num_classes,
            **_lkw,
        )

        # Carica pesi — strip prefissi, strict=False
        _clean = {}
        for _k, _v in _ckpt_state.items():
            _k2 = _k
            for _pfx in ("network.", "model.", "module."):
                while _k2.startswith(_pfx):
                    _k2 = _k2[len(_pfx):]
            _clean[_k2] = _v
        _inc = model.network.load_state_dict(_clean, strict=False)
        print(f"[MODEL][lit] missing={len(_inc.missing_keys)} unexpected={len(_inc.unexpected_keys)}")
        del _ckpt_raw, _ckpt_state, _clean

        model = model.to(device)
        model.eval()
        if model.training:
            raise RuntimeError("[FATAL] model.training=True dopo eval()")
        print(f"[MODEL] training={model.training}  (atteso: False) ✓")
        print(f"[MODEL][lit] img_size={_lit_img_size} — window_imgs_semantic scala dinamicamente")

    else:
        model = EoMTWrapper(
            img_size=size_hw,
            num_classes=args.num_classes,
            num_q=100,
            num_blocks=3,
            backbone_name=backbone,
            masked_attn_enabled=True,
        )
        model.load(args.ckpt, device)

    # PATCH — eval mode sul wrapper
    # EoMTWrapper.load() chiama .eval() solo su self.net (il modello interno),
    # ma lascia il wrapper stesso (model) in training=True.
    # BatchNorm e Dropout rimangono attivi → output stocastici e sbagliati.
    # La chiamata esplicita qui mette in eval l'intero albero di moduli.
    model.eval()

    # --- Verifica che il modello sia effettivamente in eval e su device ---
    if model.training:
        raise RuntimeError(
            "[FATAL] model.training=True dopo model.eval() — "
            "controllare EoMTWrapper o submoduli con .train() forzato."
        )
    print(f"[MODEL] training={model.training}  (atteso: False) ✓")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] parametri totali={total_params:,}")

    # --- Trova immagini ---
    paths = sorted(glob.glob(os.path.expanduser(args.input)))
    if not paths:
        raise FileNotFoundError(f"Nessuna immagine trovata: {args.input}")
    print(f"[DATA] immagini trovate: {len(paths)}")
    print(f"[DATA] prima: {paths[0]}")
    print(f"[DATA] ultima: {paths[-1]}")

    anomaly_list: List[np.ndarray] = []
    ood_list:     List[np.ndarray] = []
    names:        List[str]        = []
    mask_logits_cache:  List[np.ndarray] = []
    class_logits_cache: List[np.ndarray] = []
    pixel_logits_cache: List[np.ndarray] = []
    logits_h = logits_w = None
    unique_before_all: set = set()
    unique_after_all:  set = set()

    n_skipped_no_gt   = 0
    n_skipped_no_ood  = 0
    n_processed       = 0

    # Per debug: stampa info dettagliate solo sulla prima immagine elaborata
    _first_image_done = False

    for img_idx, p in enumerate(paths):
        img_pil = Image.open(p).convert("RGB")
        orig_hw = (img_pil.height, img_pil.width)

        # --- Caricamento GT ---
        try:
            path_gt = gt_path_from_image(p)
            raw = np.array(Image.open(path_gt))
            unique_before_all.update(int(x) for x in np.unique(raw).tolist())

            gt_size = orig_hw if args.sliding_window else size_hw
            ood = load_ood_mask(p, size_hw=gt_size)
            unique_after_all.update(int(x) for x in np.unique(ood).tolist())

            if args.debug and not _first_image_done:
                print(f"\n[DBG:GT] prima immagine: {os.path.basename(p)}")
                print(f"  orig_hw={orig_hw}  gt_size={gt_size}")
                print(f"  path_gt={path_gt}")
                _dbg_array("raw GT (pre-remap)", raw)
                _dbg_array("GT dopo remap", ood)

        except Exception as e:
            print(f"[SKIP:GT_ERROR] {os.path.basename(p)}: {e}")
            n_skipped_no_gt += 1
            continue

        if 1 not in np.unique(ood):
            if args.debug:
                print(f"[SKIP:NO_OOD] {os.path.basename(p)}  unique_after={np.unique(ood).tolist()}")
            n_skipped_no_ood += 1
            continue

        # Costruzione tensore input
        if args.no_totensor:
            # Replica gruppo 5: PIL → np.array uint8 [0,255] → tensor float [0,255]
            if input_transform is not None:
                img_resized = input_transform(img_pil)  # solo Resize, no ToTensor
            else:
                img_resized = img_pil  # SW mode: nessun resize, modello gestisce internamente
            import numpy as _np
            x = torch.from_numpy(_np.array(img_resized)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            x = input_transform(img_pil).unsqueeze(0).float().to(device)

        if args.debug and not _first_image_done:
            print(f"  input tensor: shape={tuple(x.shape)} min={x.min():.4f} max={x.max():.4f}")

        # ── Standard inference ────────────────────────────────────────────────
        if not args.sliding_window:
            mask_logits, class_logits = model.forward_masks_and_classes(x)

            if logits_h is None:
                logits_h, logits_w = int(mask_logits.shape[-2]), int(mask_logits.shape[-1])
                print(f"[LOGITS] prima shape mask_logits={tuple(mask_logits.shape)}  "
                      f"class_logits={tuple(class_logits.shape)}")

            if args.debug and not _first_image_done:
                _dbg_tensor("mask_logits", mask_logits)
                _dbg_tensor("class_logits", class_logits)

            if args.method == "rba":
                anomaly = rba_from_masks(mask_logits, class_logits, args.num_classes, args.temperature)
            elif args.method == "maxlogit":
                anomaly = anomaly_maxlogit_from_masks(mask_logits, class_logits, args.num_classes)
            else:
                pixel_probs = pixel_probs_from_masks(mask_logits, class_logits, args.num_classes, args.temperature)
                if args.debug and not _first_image_done:
                    _dbg_tensor("pixel_probs", pixel_probs)
                anomaly = anomaly_from_pixel_probs(pixel_probs, args.method)

            if anomaly.shape[-2:] != size_hw:
                print(f"  [WARN:STD] anomaly shape {tuple(anomaly.shape[-2:])} != size_hw {size_hw} "
                      f"— interpolate applicato su {os.path.basename(p)}")
                anomaly = F.interpolate(
                    anomaly.unsqueeze(1), size=size_hw, mode="bilinear", align_corners=False
                ).squeeze(1)

            if args.save_logits:
                mask_logits_cache.append(mask_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())
                class_logits_cache.append(class_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())

        # ── Sliding window inference ──────────────────────────────────────────
        else:
            # ── BRANCH B: Lightning window_imgs_semantic (gruppo 5) ──────────
            if args.lightning_sw:
                # Replica esatta di semantic_inference() del gruppo 5.
                # window_imgs_semantic si aspetta un tensore uint8 [0,255]
                # perché internamente fa Image.fromarray(img.permute(1,2,0).numpy()).
                # Convertiamo da float [0,1] a uint8 [0,255].
                img_tensor = (x.squeeze(0).cpu() * 255).to(torch.uint8).to(device)  # [3,H,W] uint8
                imgs = [img_tensor]
                img_sizes = [img_tensor.shape[-2:]]

                _ac_dtype  = torch.float16 if device.type == "cuda" else torch.bfloat16
                with torch.autocast(dtype=_ac_dtype, device_type=device.type):
                    crops_lit, origins_lit = model.window_imgs_semantic(imgs)
                    n_crops = len(crops_lit)

                    if logits_h is None:
                        logits_h, logits_w = H, W
                        print(f"[SW:LIT] prima immagine: orig_hw={orig_hw} "
                              f"n_crops={n_crops} origins={origins_lit}")

                    mask_logits_layers, class_logits_layers = model.network(crops_lit)
                    mask_logits_lit = F.interpolate(
                        mask_logits_layers[-1], model.img_size, mode="bilinear"
                    )
                    crop_logits_lit = model.to_per_pixel_logits_semantic(
                        mask_logits_lit, class_logits_layers[-1]
                    )
                    logits_lit = model.revert_window_logits_semantic(
                        crop_logits_lit, origins_lit, img_sizes
                    )
                    pixel_logits = logits_lit[0].float()  # [C, H, W]

                if args.debug and not _first_image_done:
                    _dbg_tensor("pixel_logits finali (Lightning SW)", pixel_logits)
                    pl_max = pixel_logits.max().item()
                    print(f"  [DBG:LIT] pixel_logits max={pl_max:.4f}")

                anomaly = _anomaly_from_pixel_logits(
                    pixel_logits,
                    method=args.method,
                    temperature=args.temperature,
                ).unsqueeze(0)

                anomaly_hw = tuple(anomaly.shape[-2:])
                gt_hw      = tuple(ood.shape[-2:])
                if anomaly_hw != gt_hw:
                    print(f"  [WARN:LIT_SHAPE] mismatch anomaly={anomaly_hw} gt={gt_hw} "
                          f"su {os.path.basename(p)}")
                    anomaly = F.interpolate(
                        anomaly.unsqueeze(0), size=gt_hw,
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)

                print(f"  [SW:LIT] {os.path.basename(p)}: crops={n_crops} "
                      f"orig={orig_hw} anomaly={tuple(anomaly.shape[-2:])} gt={gt_hw}")

                if args.save_logits:
                    pixel_logits_cache.append(
                        pixel_logits.detach().cpu().to(torch.float16).numpy()
                    )

            # ── BRANCH A: SlidingWindow custom (default) ─────────────────────
            else:
                sw = SlidingWindow(img_size=min(H, W), device=device)
                crops, origins, _ = sw.window_image(x)
                n_crops = len(crops)

                if logits_h is None:
                    logits_h, logits_w = H, W
                    print(f"[SW] prima immagine: orig_hw={orig_hw} n_crops={n_crops} "
                          f"crop_size={min(H,W)}x{min(H,W)}")
                    print(f"[SW] origins: {origins}")

                crop_idx = 0
                _ac_dtype  = torch.float16 if device.type == "cuda" else torch.bfloat16
                with torch.autocast(dtype=_ac_dtype, device_type=device.type):
                    for batch in sw.iter_batches(crops, batch_size=args.sw_batch_size):
                        batch = batch.to(device)
                        ml, cl = model.forward_masks_and_classes(batch)

                        if args.debug and not _first_image_done and crop_idx == 0:
                            print(f"\n[DBG:SW] crop 0 — batch shape={tuple(batch.shape)}")
                            _dbg_tensor("mask_logits (crop 0)", ml)
                            _dbg_tensor("class_logits (crop 0)", cl)

                        pl = SlidingWindow.to_pixel_logits(ml, cl, args.num_classes)

                        if args.debug and not _first_image_done and crop_idx == 0:
                            _dbg_tensor("pixel_logits dopo to_pixel_logits (crop 0)", pl)

                        indices = list(range(crop_idx, crop_idx + batch.shape[0]))
                        sw.accumulate(pl.float(), origins, orig_hw, indices)
                        crop_idx += batch.shape[0]

                pixel_logits = sw.finalize(orig_hw)  # [C, H, W]

                if args.debug and not _first_image_done:
                    _dbg_tensor("pixel_logits finali (dopo finalize, float32)", pixel_logits)
                    pl_max = pixel_logits.max().item()
                    pl_mean = pixel_logits.mean().item()
                    if pl_max > 5.0:
                        print(f"  [DBG:AUTOCAST] pixel_logits max={pl_max:.4f} — "
                              f"alto, autocast potrebbe non essere attivo o efficace")
                    else:
                        print(f"  [DBG:AUTOCAST] pixel_logits max={pl_max:.4f} mean={pl_mean:.4f} — "
                              f"range compresso, autocast float16 attivo ✓")

                anomaly = _anomaly_from_pixel_logits(
                    pixel_logits,
                    method=args.method,
                    temperature=args.temperature,
                ).unsqueeze(0)  # [1, H, W]

                anomaly_hw = tuple(anomaly.shape[-2:])
                gt_hw      = tuple(ood.shape[-2:])
                if anomaly_hw != gt_hw:
                    print(f"  [WARN:SW_SHAPE] mismatch anomaly={anomaly_hw} gt={gt_hw} "
                          f"su {os.path.basename(p)} — bilinear upsample applicato")
                    anomaly = F.interpolate(
                        anomaly.unsqueeze(0),
                        size=gt_hw,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                print(f"  [SW] {os.path.basename(p)}: crops={n_crops} orig={orig_hw} "
                      f"anomaly={tuple(anomaly.shape[-2:])} gt={gt_hw}")

                if args.save_logits:
                    pixel_logits_cache.append(
                        pixel_logits.detach().cpu().to(torch.float16).numpy()
                    )

        # --- Debug anomaly scores sulla prima immagine elaborata ---
        if args.debug and not _first_image_done:
            _dbg_tensor("anomaly map (pre-squeeze)", anomaly)
            anomaly_np = anomaly.squeeze(0).detach().cpu().float().numpy()
            _dbg_anomaly_score_distribution(
                f"prima immagine [{os.path.basename(p)}]",
                anomaly_np,
                ood
            )
            _first_image_done = True

        anomaly_list.append(anomaly.squeeze(0).detach().cpu().float().numpy())
        ood_list.append(ood)
        names.append(os.path.basename(p))
        n_processed += 1

    # --- Statistiche di avanzamento ---
    print(f"\n[SUMMARY] immagini totali:   {len(paths)}")
    print(f"[SUMMARY] elaborate:         {n_processed}")
    print(f"[SUMMARY] skipped no GT:     {n_skipped_no_gt}")
    print(f"[SUMMARY] skipped no OOD px: {n_skipped_no_ood}")
    print(f"[SUMMARY] mask_unique_before={sorted(unique_before_all)}")
    print(f"[SUMMARY] mask_unique_after= {sorted(unique_after_all)}")

    if n_processed == 0:
        raise RuntimeError(
            "Nessuna immagine elaborata. "
            f"mask_unique_before={sorted(unique_before_all)} "
            f"mask_unique_after={sorted(unique_after_all)}"
        )

    ood_gts        = np.array(ood_list)
    anomaly_scores = np.array(anomaly_list)

    # --- Debug distribuzione score globale prima del calcolo metriche ---
    if args.debug:
        print(f"\n[DBG:GLOBAL] distribuzione score globale su tutto il dataset:")
        _dbg_anomaly_score_distribution("GLOBAL", anomaly_scores.ravel(), ood_gts.ravel())
        print(f"[DBG:GLOBAL] ood_gts shape={ood_gts.shape}  "
              f"unique={np.unique(ood_gts).tolist()}")
        print(f"[DBG:GLOBAL] anomaly_scores shape={anomaly_scores.shape}  "
              f"nan={np.isnan(anomaly_scores).sum()}  "
              f"inf={np.isinf(anomaly_scores).sum()}")

    ood_out = anomaly_scores[ood_gts == 1]
    in_out  = anomaly_scores[ood_gts == 0]

    print(f"[METRICS] pixel OOD usati: {len(ood_out):,}  InD usati: {len(in_out):,}  "
          f"ignored: {int((ood_gts == 255).sum()):,}")

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
        "images_used":         int(n_processed),
        "images_skipped_gt":   int(n_skipped_no_gt),
        "images_skipped_ood":  int(n_skipped_no_ood),
        "pixels_ood":          int(len(ood_out)),
        "pixels_ind":          int(len(in_out)),
        "device":              str(device),
        "cudnn_benchmark":     bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "mask_unique_before":  sorted(unique_before_all),
        "mask_unique_after":   sorted(unique_after_all),
    }

    print("=" * 60)
    print(f"RISULTATI FINALI")
    print(f"  dataset={args.dataset_name}  method={args.method}  T={args.temperature}")
    print(f"  sw={args.sliding_window}  ckpt={ckpt_basename}  sha1={ckpt_sha1_8}")
    print(f"  AUPRC:    {metrics['auprc_pct']:.4f}%")
    print(f"  FPR@95:   {metrics['fpr95_pct']:.4f}%")
    print(f"  immagini: {n_processed}  (skip_gt={n_skipped_no_gt} skip_ood={n_skipped_no_ood})")
    print(f"  pixel OOD={len(ood_out):,}  InD={len(in_out):,}")
    print(f"  crop: {H}x{W}  logits_ref: {logits_h}x{logits_w}")
    print("=" * 60)

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
        print(f"[CACHED] mask_logits / class_logits / gt / names → {art.logits}")

    if args.save_logits and args.sliding_window:
        ds = args.dataset_name
        np.save(art.logits / f"{ds}__pixel_logits_f16.npy", np.array(pixel_logits_cache, dtype=np.float16))
        np.save(art.logits / f"{ds}__gt.npy",               ood_gts.astype(np.uint8))
        with open(art.logits / f"{ds}__names.json", "w", encoding="utf-8") as f:
            json.dump(names, f, indent=2)
        print(f"[CACHED] pixel_logits / gt / names → {art.logits}")


if __name__ == "__main__":
    main()