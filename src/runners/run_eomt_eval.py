# src/runners/run_eomt_eval.py
#
# EoMT inference runner for anomaly segmentation.
#
# This script evaluates an EoMT checkpoint on an OOD validation dataset and
# produces per-image anomaly maps that are scored with post-hoc methods
# (MSP, MaxLogit, MaxEntropy, RbA). Results and (optionally) cached logits
# are written into a per-run artifact directory.
#
# The runner supports four operating modes ("branches"), selected via CLI flags.
# Branches differ only in HOW the model is built and HOW the sliding-window
# inference is performed. The post-hoc scoring stage is shared.
#
# ┌──────────────────┬───────────────────────────────────────────────────────┐
# │ Branch           │ How it works                                          │
# ├──────────────────┼───────────────────────────────────────────────────────┤
# │ A (default)      │ Custom EoMTWrapper at a fixed square resolution       │
# │                  │ resolved from --resize / --inference-size / config    │
# │                  │ name. Crops are produced by the custom SlidingWindow  │
# │                  │ utility.                                              │
# │                  │                                                       │
# │ A-interp         │ Same as A, but at load time the pos_embed of the      │
# │ --interp-pos-    │ checkpoint is bicubically resized to the model's      │
# │   embed          │ grid instead of being silently discarded by strict=   │
# │                  │ False. Useful when the checkpoint was trained at a    │
# │                  │ resolution different from the one the model is        │
# │                  │ instantiated with.                                    │
# │                  │                                                       │
# │ B                │ Uses the MaskClassificationSemantic Lightning module  │
# │ --lightning-sw   │ together with window_imgs_semantic / revert_window_   │
# │                  │ logits_semantic for sliding-window inference. Inputs  │
# │                  │ are passed as uint8 [0,255] tensors so the Lightning  │
# │                  │ forward divides them by 255 internally and applies    │
# │                  │ ImageNet normalization.                               │
# │                  │                                                       │
# │ C                │ Same Lightning-based SW as branch B, but uses the     │
# │ --lightning-v2   │ direct PIL Resize(512,1024) + uint8 conversion path,  │
# │                  │ loads weights straight into the wrapper without       │
# │                  │ stripping prefixes, and supports an explicit          │
# │                  │ --inference-size override of the resolution that      │
# │                  │ would otherwise be auto-detected from pos_embed.      │
# └──────────────────┴───────────────────────────────────────────────────────┘
#
# The --debug flag enables per-image and global diagnostic prints (tensor
# shapes, value ranges, NaN/Inf counts, and InD-vs-OOD score distributions),
# which are very useful when investigating regressions. It is off by default
# because it can produce a lot of output.

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
    """
    Print a one-line summary of a tensor: shape, dtype, device, min/mean/max,
    plus NaN and Inf counts. Used only when --debug is set.
    """
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
    """
    Print shape, dtype, and the set of unique values of a numpy array,
    typically a ground-truth mask.
    """
    print(
        f"  [DBG] {tag}: shape={a.shape} dtype={a.dtype} "
        f"unique={np.unique(a).tolist()}"
    )


def _dbg_anomaly_score_distribution(tag: str, scores: np.ndarray, gt: np.ndarray) -> None:
    """
    Print the score distribution split between in-distribution pixels (gt==0)
    and out-of-distribution pixels (gt==1), along with the count of ignored
    pixels (gt==255). Also reports the score direction, which should always
    have OOD > InD on the median. A reversed direction is a strong signal
    that the anomaly score sign is wrong or that the model output collapsed.
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
        # OOD median should sit above InD median. If not, anomaly score sign
        # is wrong or the model output has collapsed to a constant.
        direction = (
            "OK (OOD > InD)"
            if np.median(ood_scores) > np.median(ind_scores)
            else "!! INVERTED (OOD < InD) !!"
        )
        print(f"    score direction: {direction}")


# ---------------------------------------------------------------------------
# Other helpers
# ---------------------------------------------------------------------------

def _sha1_8_of_file(path: str) -> str:
    """Short SHA1 prefix of a file, used to disambiguate checkpoint artifacts."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def append_metrics_csv(csv_path: Path, row: dict) -> None:
    """Append one row to a metrics CSV, writing the header on first creation."""
    write_header = not csv_path.exists()
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Sliding-window anomaly scoring
# ---------------------------------------------------------------------------
#
# In sliding-window mode the per-pixel logits returned by
# SlidingWindow.to_pixel_logits() are produced by
#
#     pixel_logits = sigmoid(mask_logits) @ softmax(class_logits)
#
# so they live in roughly [0, 1] rather than in the unbounded logit space.
# This affects how each post-hoc method should be computed:
#
#   * MSP        -> 1 - max(softmax(pixel_logits / T))
#   * MaxLogit   -> -max(pixel_logits) (consistent ranking with the non-SW
#                   path, where logits are negated to get an anomaly score)
#   * MaxEntropy -> sum(-p * log p), p = softmax(pixel_logits / T)
#   * RbA        -> -sum(tanh(pixel_logits))
#                   Note: pixel_logits already lie in [0, 1], so tanh is
#                   nearly linear here and the variance is compressed. This
#                   is a documented structural limitation of RbA in SW mode.

def _anomaly_from_pixel_logits(
    pixel_logits: torch.Tensor,
    method: str,
    temperature: float,
) -> torch.Tensor:
    """
    Compute an anomaly map [H, W] from per-pixel logits of shape [C, H, W].
    """
    pl = pixel_logits.unsqueeze(0)  # [1, C, H, W]

    if method == "maxlogit":
        # Negative max, so the score is "anomalous" when the maximum logit
        # over classes is low (model is uncertain).
        return (-pl.max(dim=1).values).squeeze(0)

    if method == "msp":
        probs = (pl / temperature).softmax(dim=1)
        return (1.0 - probs.max(dim=1).values).squeeze(0)

    if method == "maxentropy":
        probs = (pl / temperature).softmax(dim=1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
        return entropy.squeeze(0)

    if method == "rba":
        # Reference formula. Applied to SW pixel_logits already in [0, 1] it
        # produces a compressed distribution, so absolute AUPRC values may
        # look low even on easy datasets.
        return -torch.tanh(pl).sum(dim=1).squeeze(0)

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    # ── I/O ──────────────────────────────────────────────────────────────
    ap.add_argument("--input",        required=True,
                    help="Glob pattern matching the input RGB images.")
    ap.add_argument("--ckpt",         required=True,
                    help="Path to the EoMT checkpoint (.bin).")
    ap.add_argument("--config",       default="eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
                    help="EoMT YAML configuration file. Used by the Lightning "
                         "branches (B, C) to build the encoder / network / "
                         "Lightning module via class_path import.")
    ap.add_argument("--dataset-name", required=True,
                    help="Logical dataset name, used as a sub-folder under "
                         "--artifacts-dir.")

    # ── Scoring and model setup ──────────────────────────────────────────
    ap.add_argument("--method",      choices=["msp", "maxlogit", "maxentropy", "rba"], default="msp",
                    help="Post-hoc anomaly scoring method.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Temperature applied before softmax (MSP, MaxEntropy).")
    ap.add_argument("--num-classes", type=int,   default=19,
                    help="Number of foreground classes. Cityscapes = 19, "
                         "COCO panoptic = 133.")
    ap.add_argument("--resize",      default=None,
                    help="Override the inference resolution as HxW (legacy).")
    ap.add_argument("--inference-size", type=int, default=None,
                    help="Square inference resolution N x N. For the lightning_v2 "
                         "branch this overrides the resolution that would "
                         "otherwise be derived from the checkpoint's pos_embed.")
    ap.add_argument("--mode",        choices=["robust", "prof-exact"], default="robust",
                    help="Determinism strategy.")

    # ── Sliding-window options ───────────────────────────────────────────
    ap.add_argument("--sliding-window", action="store_true",
                    help="Run inference in sliding-window mode.")
    ap.add_argument("--sw-batch-size",  type=int, default=1,
                    help="How many crops to forward per micro-batch in SW mode.")

    # ── Determinism and artifacts ────────────────────────────────────────
    ap.add_argument("--seed",          type=int,  default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--save-logits",   action="store_true",
                    help="Cache the recomposed per-image logits to disk for "
                         "post-hoc temperature sweeps.")
    ap.add_argument("--cpu",           action="store_true")

    # ── Debug and branch selection ───────────────────────────────────────
    ap.add_argument("--debug", action="store_true",
                    help="Enable verbose diagnostics: per-image tensor stats, "
                         "GT mask inspection, and InD vs OOD score "
                         "distributions on the first image and globally.")
    ap.add_argument("--no-totensor", action="store_true",
                    help="Skip the /255 normalization and feed the model with "
                         "uint8 [0,255] tensors. Experimental, only useful for "
                         "comparing different preprocessing pipelines.")
    ap.add_argument("--lightning-sw", action="store_true",
                    help="Branch B: build a MaskClassificationSemantic Lightning "
                         "module and perform sliding-window inference using its "
                         "window_imgs_semantic / revert_window_logits_semantic "
                         "methods. The state dict is loaded after stripping the "
                         "'network.' / 'model.' / 'module.' prefixes.")
    ap.add_argument("--lightning-v2", action="store_true",
                    help="Branch C: same Lightning-based sliding-window inference "
                         "as branch B, but the state dict is loaded directly into "
                         "the Lightning wrapper (no prefix stripping), the input "
                         "is built with PIL Resize(512,1024) + uint8 conversion, "
                         "and --inference-size can override the resolution "
                         "auto-detected from pos_embed.")
    ap.add_argument("--interp-pos-embed", action="store_true",
                    help="Branch A-interp: when loading the checkpoint into "
                         "EoMTWrapper, bicubically resize the pos_embed to the "
                         "model's grid instead of letting strict=False drop it "
                         "silently. Has no effect with --lightning-sw or "
                         "--lightning-v2.")

    args = ap.parse_args()

    if args.sliding_window and args.save_logits:
        print("[INFO] --save-logits in SW mode: pixel logits [N,C,H,W] will be cached.")

    # ── Determinism ──────────────────────────────────────────────────────
    want_determinism = (args.mode == "robust") or bool(args.deterministic)
    apply_determinism(mode=args.mode, seed=int(args.seed), deterministic=bool(want_determinism))

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Resolve inference resolution (used by branches A and A-interp) ───
    # Priority:
    #   1. --resize HxW (explicit, legacy)
    #   2. --inference-size N (square, explicit)
    #   3. heuristic on the config file name (legacy default)
    #
    # The lightning_v2 branch (C) ignores this value and derives its own
    # resolution from the checkpoint's pos_embed; --inference-size, if
    # provided, overrides that derived value inside branch C as well.
    if args.resize is not None:
        hw = args.resize.lower().replace(" ", "").split("x")
        if len(hw) != 2:
            raise ValueError("--resize must be formatted as HxW, e.g. 640x640")
        H, W = int(hw[0]), int(hw[1])
    elif args.inference_size is not None:
        H = W = int(args.inference_size)
    else:
        cfg_lower = os.path.basename(args.config).lower()
        H = W = 1024 if "1024" in cfg_lower else 640

    size_hw = (H, W)

    # ── Build the input transform ────────────────────────────────────────
    # In standard (non-SW) mode the image is resized to size_hw and converted
    # to a float tensor in [0, 1]. In SW mode resizing is delegated to the
    # sliding-window utilities, so we only apply ToTensor() to keep tensors
    # consistent.
    #
    # The --no-totensor flag skips the /255 normalization and passes uint8
    # [0, 255] arrays through as floats. The EoMT backbone expects [0, 1]
    # ImageNet-normalized inputs, so this option exists only to compare
    # against alternative preprocessing pipelines and is not used in the
    # default evaluation flow.
    if not args.sliding_window:
        if args.no_totensor:
            input_transform = Compose([Resize(size_hw, Image.BILINEAR)])
        else:
            input_transform = Compose([Resize(size_hw, Image.BILINEAR), ToTensor()])
    else:
        if args.no_totensor:
            input_transform = None  # handled inline in the SW loop
        else:
            input_transform = ToTensor()

    # ── Session header ───────────────────────────────────────────────────
    print("=" * 60)
    print(f"[SESSION] dataset={args.dataset_name} method={args.method} T={args.temperature}")
    print(f"[SESSION] mode={args.mode} sw={args.sliding_window} sw_batch={args.sw_batch_size}")
    print(f"[SESSION] ckpt={args.ckpt}")
    print(f"[SESSION] config={args.config}")
    print(f"[SESSION] resize={H}x{W} num_classes={args.num_classes}")
    print(f"[SESSION] device={device} seed={args.seed} deterministic={want_determinism}")
    print(f"[SESSION] debug={args.debug}")
    print(f"[SESSION] no_totensor={args.no_totensor}  (feed uint8 [0,255] to the model)")
    print(f"[SESSION] lightning_sw={args.lightning_sw}  (branch B)")
    print(f"[SESSION] lightning_v2={args.lightning_v2}  (branch C)")
    print(f"[SESSION] interp_pos_embed={args.interp_pos_embed}  (branch A-interp)")
    print(f"[SESSION] inference_size={args.inference_size}  (explicit square override)")
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

    # ── Backbone selection ───────────────────────────────────────────────
    # Branch A instantiates the backbone directly through EoMTWrapper, and
    # picks vit_base or vit_large based on the config file name.
    backbone = (
        "vit_large_patch14_reg4_dinov2"
        if "large" in os.path.basename(args.config).lower()
        else "vit_base_patch14_reg4_dinov2"
    )
    print(f"[MODEL] backbone={backbone}")

    # ── Model construction ───────────────────────────────────────────────
    # Three mutually exclusive paths:
    #   * Branch B (--lightning-sw): Lightning module, fuzzy load with prefix
    #     stripping.
    #   * Branch C (--lightning-v2): Lightning module, direct load on the
    #     wrapper, optional --inference-size override.
    #   * Branch A (default): custom EoMTWrapper, optionally with pos_embed
    #     bicubic resize (--interp-pos-embed → "A-interp").

    if args.lightning_sw:
        # ── Branch B ────────────────────────────────────────────────────
        # Instantiate MaskClassificationSemantic with the img_size detected
        # from the checkpoint's pos_embed (e.g. 1024x1024 when the checkpoint
        # was trained at that resolution), then load the state dict on the
        # Lightning module itself. window_imgs_semantic is later used to
        # build square crops at that resolution.
        print("[MODEL] Loading via MaskClassificationSemantic (branch B) ...")
        import importlib, yaml, sys as _sys

        # Ensure the local eomt/ package is importable when this runner is
        # executed from the repo root.
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
            print("[MODEL][lit] lightning installed ✓")

        # Inspect pos_embed to recover the resolution the checkpoint was
        # trained at. The grid size determines img_size as grid * patch.
        _ckpt_raw = torch.load(args.ckpt, map_location="cpu")
        _ckpt_state = _ckpt_raw.get("state_dict", _ckpt_raw)
        _lit_img_size = (1024, 1024)
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

        # Build the encoder at the detected img_size.
        _ec = _cfg["model"]["init_args"]["network"]["init_args"]["encoder"]
        _em, _en = _ec["class_path"].rsplit(".", 1)
        _encoder = getattr(importlib.import_module(_em), _en)(
            img_size=_lit_img_size, **_ec.get("init_args", {})
        )

        # Build the network. Masked attention is disabled for inference.
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

        # Build the Lightning module, dropping any kwargs not accepted by
        # the constructor (panoptic-specific ones in particular).
        _lm, _ln = _cfg["model"]["class_path"].rsplit(".", 1)
        _lkw = {k: v for k, v in _cfg["model"]["init_args"].items() if k != "network"}

        # Lift stuff_classes from data.init_args if the model needs it
        # (panoptic Lightning module).
        if "stuff_classes" not in _lkw:
            _data_init = _cfg.get("data", {}).get("init_args", {}) or {}
            if "stuff_classes" in _data_init:
                _lkw["stuff_classes"] = _data_init["stuff_classes"]

        for _rk in ("overlap_thresh", "mask_thresh"):
            _lkw.pop(_rk, None)

        import inspect as _inspect
        _cls = getattr(importlib.import_module(_lm), _ln)
        _sig_params = set(_inspect.signature(_cls.__init__).parameters.keys())
        if "stuff_classes" in _lkw and "stuff_classes" not in _sig_params:
            _lkw.pop("stuff_classes", None)

        model = _cls(
            network=_network,
            img_size=_lit_img_size,
            num_classes=args.num_classes,
            **_lkw,
        )

        # Load weights with prefix stripping. This is the "fuzzy" load that
        # tolerates wrappers having a different naming convention than the
        # checkpoint (e.g. "network.encoder..." vs "encoder...").
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
            raise RuntimeError("[FATAL] model.training=True after eval()")
        print(f"[MODEL] training={model.training}  (expected: False) ✓")
        print(f"[MODEL][lit] img_size={_lit_img_size} — window_imgs_semantic rescales dynamically")

    elif args.lightning_v2:
        # ── Branch C ────────────────────────────────────────────────────
        # Same Lightning-based sliding-window pipeline as branch B, with
        # three practical differences:
        #
        #   1. The state dict is loaded DIRECTLY on the Lightning module
        #      (no prefix stripping). This works when the published
        #      checkpoint already contains keys with the expected wrapper
        #      prefix.
        #
        #   2. Inputs are built with a fixed PIL Resize(512, 1024) followed
        #      by a uint8 tensor conversion. The Lightning forward divides
        #      by 255 internally, so calling model(crops) (NOT
        #      model.network(crops)) is essential — bypassing the wrapper
        #      forward skips that division and the network sees values up
        #      to 255, which collapses the output.
        #
        #   3. The resolution is normally derived from the checkpoint's
        #      pos_embed, but can be overridden via --inference-size to
        #      enable ablations such as comparing 640x640 vs 1024x1024
        #      using the same weights.
        print("[MODEL] Loading via MaskClassificationSemantic (branch C, lightning_v2) ...")
        import importlib, yaml, sys as _sys

        _eomt_dir = os.path.join(os.getcwd(), "eomt")
        if _eomt_dir not in _sys.path:
            _sys.path.insert(0, _eomt_dir)

        try:
            import lightning as _lt
            print(f"[MODEL][v2] lightning {_lt.__version__}")
        except ImportError:
            import subprocess as _sp
            _sp.run([_sys.executable, "-m", "pip", "install", "-q", "lightning"], check=True)

        # Native resolution from pos_embed (1024x1024 when seq=4096,
        # 640x640 when seq=1600).
        _ckpt_raw = torch.load(args.ckpt, map_location="cpu")
        _ckpt_state = _ckpt_raw.get("state_dict", _ckpt_raw)
        _v2_img_size = (1024, 1024)
        for _pk, _pv in _ckpt_state.items():
            if "pos_embed" in _pk and _pv.dim() == 3:
                _seq = _pv.shape[1]
                if _seq in (1600, 1605):
                    _v2_img_size = (640, 640)
                print(f"[MODEL][v2] pos_embed seq={_seq} → img_size={_v2_img_size}")
                break

        # Explicit override of the inference resolution (used for the
        # 640-vs-1024 ablation on the same checkpoint).
        if args.inference_size is not None:
            _override = (int(args.inference_size), int(args.inference_size))
            print(f"[MODEL][v2] --inference-size active: {_v2_img_size} -> {_override}")
            _v2_img_size = _override

        with open(args.config) as _f:
            _cfg = yaml.safe_load(_f)

        _ec = _cfg["model"]["init_args"]["network"]["init_args"]["encoder"]
        _em, _en = _ec["class_path"].rsplit(".", 1)
        _encoder = getattr(importlib.import_module(_em), _en)(
            img_size=_v2_img_size, **_ec.get("init_args", {})
        )
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
        _lm, _ln = _cfg["model"]["class_path"].rsplit(".", 1)
        _lkw = {k: v for k, v in _cfg["model"]["init_args"].items() if k != "network"}

        # The panoptic Lightning module requires `stuff_classes` as a
        # positional argument in its constructor. In the upstream config
        # this list lives under `data.init_args.stuff_classes`, not under
        # `model.init_args`. Lift it into the model kwargs when present so
        # the panoptic class can be instantiated; the semantic class
        # ignores any extra kwargs that the runner does not need.
        if "stuff_classes" not in _lkw:
            _data_init = _cfg.get("data", {}).get("init_args", {}) or {}
            if "stuff_classes" in _data_init:
                _lkw["stuff_classes"] = _data_init["stuff_classes"]
                print(f"[MODEL][v2] stuff_classes lifted from data.init_args "
                      f"({len(_lkw['stuff_classes'])} entries)")

        # Drop training-only kwargs that the constructor does not accept.
        for _rk in ("overlap_thresh", "mask_thresh"):
            _lkw.pop(_rk, None)

        # If the target class does NOT accept stuff_classes (e.g. the
        # semantic class), drop it from kwargs.
        import inspect as _inspect
        _cls = getattr(importlib.import_module(_lm), _ln)
        _sig_params = set(_inspect.signature(_cls.__init__).parameters.keys())
        if "stuff_classes" in _lkw and "stuff_classes" not in _sig_params:
            _lkw.pop("stuff_classes", None)

        model = _cls(
            network=_network,
            img_size=_v2_img_size,
            num_classes=args.num_classes,
            **_lkw,
        )

        # Direct load on the Lightning module (no prefix stripping). When
        # the checkpoint is already saved with the wrapper's naming
        # convention, this produces missing=0 / unexpected=0.
        _inc = model.load_state_dict(_ckpt_state, strict=False)
        print(f"[MODEL][v2] missing={len(_inc.missing_keys)} unexpected={len(_inc.unexpected_keys)}")
        del _ckpt_raw, _ckpt_state

        model = model.to(device)
        model.eval()
        if model.training:
            raise RuntimeError("[FATAL] model.training=True after eval()")
        print(f"[MODEL][v2] ✓  img_size={_v2_img_size}")

    else:
        # ── Branch A / A-interp ─────────────────────────────────────────
        # Custom EoMTWrapper at a fixed square resolution. With
        # --interp-pos-embed the wrapper's load() bicubically resizes the
        # checkpoint's pos_embed to the model's grid (e.g. 64x64 -> 40x40
        # when the checkpoint was trained at 1024 and we run at 640),
        # turning what would otherwise be a silently dropped tensor
        # (missing=1) into a properly loaded one (missing=0).
        model = EoMTWrapper(
            img_size=size_hw,
            num_classes=args.num_classes,
            num_q=100,
            num_blocks=3,
            backbone_name=backbone,
            masked_attn_enabled=True,
        )
        model.load(
            args.ckpt, device,
            interp_pos_embed=bool(args.interp_pos_embed),
        )

    # Move the whole module tree to eval mode. EoMTWrapper.load() already
    # calls .eval() on the inner net, but we also call it on the wrapper
    # itself so that BatchNorm and Dropout layers anywhere in the tree are
    # frozen for inference.
    model.eval()
    if model.training:
        raise RuntimeError(
            "[FATAL] model.training=True after model.eval() — "
            "check EoMTWrapper or any submodule that forces .train()."
        )
    print(f"[MODEL] training={model.training}  (expected: False) ✓")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] total parameters={total_params:,}")

    # ── Discover input images ────────────────────────────────────────────
    paths = sorted(glob.glob(os.path.expanduser(args.input)))
    if not paths:
        raise FileNotFoundError(f"No images matched the glob: {args.input}")
    print(f"[DATA] images found: {len(paths)}")
    print(f"[DATA] first: {paths[0]}")
    print(f"[DATA] last:  {paths[-1]}")

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

    # Print detailed diagnostics only on the first processed image; otherwise
    # the log explodes on large datasets.
    _first_image_done = False

    for img_idx, p in enumerate(paths):
        img_pil = Image.open(p).convert("RGB")
        orig_hw = (img_pil.height, img_pil.width)

        # ── Load the ground-truth mask ──────────────────────────────────
        try:
            path_gt = gt_path_from_image(p)
            raw = np.array(Image.open(path_gt))
            unique_before_all.update(int(x) for x in np.unique(raw).tolist())

            # In SW mode the GT stays at the original resolution because
            # anomaly maps are recomposed at orig_hw. In standard mode the
            # GT is resized to size_hw to match the model output.
            gt_size = orig_hw if args.sliding_window else size_hw
            ood = load_ood_mask(p, size_hw=gt_size)
            unique_after_all.update(int(x) for x in np.unique(ood).tolist())

            if args.debug and not _first_image_done:
                print(f"\n[DBG:GT] first image: {os.path.basename(p)}")
                print(f"  orig_hw={orig_hw}  gt_size={gt_size}")
                print(f"  path_gt={path_gt}")
                _dbg_array("raw GT (pre-remap)", raw)
                _dbg_array("GT after remap", ood)

        except Exception as e:
            print(f"[SKIP:GT_ERROR] {os.path.basename(p)}: {e}")
            n_skipped_no_gt += 1
            continue

        # Skip images that contain no OOD pixels: they cannot contribute to
        # the OOD class statistics and would only inflate the InD set.
        if 1 not in np.unique(ood):
            if args.debug:
                print(f"[SKIP:NO_OOD] {os.path.basename(p)}  unique_after={np.unique(ood).tolist()}")
            n_skipped_no_ood += 1
            continue

        # ── Build the input tensor ──────────────────────────────────────
        # Branch C builds its own input inline (PIL resize + uint8). All
        # other branches go through the input_transform pipeline.
        if args.lightning_v2:
            # Branch C inline preprocessing. The actual model input is the
            # uint8 tensor built later in the SW block; x here is only used
            # to mirror the shape expected by downstream code.
            import numpy as _np2
            _img_resized_v2 = Image.fromarray(
                _np2.array(img_pil.resize((1024, 512), Image.BILINEAR))
            )
            x_v2 = torch.from_numpy(_np2.array(_img_resized_v2)).permute(2, 0, 1)
            x = x_v2.unsqueeze(0).float().to(device)
        elif args.no_totensor:
            if input_transform is not None:
                img_resized = input_transform(img_pil)
            else:
                img_resized = img_pil
            import numpy as _np
            x = torch.from_numpy(_np.array(img_resized)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            x = input_transform(img_pil).unsqueeze(0).float().to(device)

        if args.debug and not _first_image_done:
            print(f"  input tensor: shape={tuple(x.shape)} min={x.min():.4f} max={x.max():.4f}")

        # ── Standard (non-SW) inference ─────────────────────────────────
        if not args.sliding_window:
            mask_logits, class_logits = model.forward_masks_and_classes(x)

            if logits_h is None:
                logits_h, logits_w = int(mask_logits.shape[-2]), int(mask_logits.shape[-1])
                print(f"[LOGITS] first shape mask_logits={tuple(mask_logits.shape)}  "
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
                      f"— bilinear upsample applied on {os.path.basename(p)}")
                anomaly = F.interpolate(
                    anomaly.unsqueeze(1), size=size_hw, mode="bilinear", align_corners=False
                ).squeeze(1)

            if args.save_logits:
                mask_logits_cache.append(mask_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())
                class_logits_cache.append(class_logits.squeeze(0).detach().cpu().to(torch.float16).numpy())

        # ── Sliding-window inference ────────────────────────────────────
        else:
            # ── Branch C: lightning_v2 ──────────────────────────────────
            if args.lightning_v2:
                # The image goes through PIL Resize(512, 1024) and is then
                # converted to a uint8 tensor on the target device.
                # window_imgs_semantic upscales the short side to the model's
                # img_size and produces overlapping square crops.
                #
                # Inside the autocast block we call model(crops), which
                # routes through the Lightning forward and applies the /255
                # normalization. Calling model.network(crops) directly
                # bypasses that normalization and yields collapsed outputs.
                import numpy as _npv2
                _img_pil_512 = img_pil.resize((1024, 512), Image.BILINEAR)
                img_tensor_v2 = torch.from_numpy(
                    _npv2.array(_img_pil_512)
                ).permute(2, 0, 1).to(device)  # uint8 [3, 512, 1024]

                _v2_target = _v2_img_size
                model.window_size = _v2_target[0]

                img_sizes_v2 = [img_tensor_v2.shape[-2:]]

                _ac_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
                with torch.autocast(dtype=_ac_dtype, device_type=device.type):
                    crops_v2, origins_v2 = model.window_imgs_semantic([img_tensor_v2])
                    n_crops = len(crops_v2)

                    if logits_h is None:
                        logits_h, logits_w = _v2_target
                        print(f"[SW:V2] first image: orig_hw={orig_hw} "
                              f"n_crops={n_crops} origins={origins_v2}")

                    # Lightning forward, which applies /255 and ImageNet
                    # normalization internally before calling self.network.
                    ml_v2, cl_v2 = model(crops_v2)
                    ml_v2_interp = F.interpolate(
                        ml_v2[-1], _v2_target, mode="bilinear"
                    )
                    crop_logits_v2 = model.to_per_pixel_logits_semantic(
                        ml_v2_interp, cl_v2[-1]
                    )
                    logits_v2 = model.revert_window_logits_semantic(
                        crop_logits_v2, origins_v2, img_sizes_v2
                    )
                    # Shape: [C, H_scaled, W_scaled].
                    pixel_logits_v2 = logits_v2[0].float()

                if args.debug and not _first_image_done:
                    _dbg_tensor("pixel_logits_v2", pixel_logits_v2)
                    print(f"  [DBG:V2] pixel_logits max={pixel_logits_v2.max().item():.4f} "
                          f"mean={pixel_logits_v2.mean().item():.4f}")

                anomaly = _anomaly_from_pixel_logits(
                    pixel_logits_v2,
                    method=args.method,
                    temperature=args.temperature,
                ).unsqueeze(0)

                # Upsample the anomaly map to the original GT resolution
                # before scoring, so that AUPRC / FPR95 are computed over
                # the same pixel grid as the ground truth.
                anomaly_hw = tuple(anomaly.shape[-2:])
                gt_hw      = tuple(ood.shape[-2:])
                if anomaly_hw != gt_hw:
                    anomaly = F.interpolate(
                        anomaly.unsqueeze(0), size=gt_hw,
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)

                print(f"  [SW:V2] {os.path.basename(p)}: crops={n_crops} "
                      f"orig={orig_hw} anomaly={tuple(anomaly.shape[-2:])} gt={gt_hw}")

                if args.save_logits:
                    pixel_logits_cache.append(
                        pixel_logits_v2.detach().cpu().to(torch.float16).numpy()
                    )

            # ── Branch B: lightning_sw ──────────────────────────────────
            elif args.lightning_sw:
                # The Lightning sliding-window utilities expect uint8 tensors
                # in [0, 255]. We convert x (which is float [0, 1] from
                # ToTensor()) back to uint8 to match that contract.
                #
                # The autocast block wraps the entire SW pipeline so that
                # window_imgs_semantic, model forward, and the per-pixel
                # logit conversion all run in float16. model.window_size is
                # set explicitly to the detected img_size so crops have the
                # expected shape.
                img_tensor = (x.squeeze(0).cpu() * 255).to(torch.uint8).to(device)
                imgs = [img_tensor]
                img_sizes = [img_tensor.shape[-2:]]

                _ac_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
                model.window_size = _lit_img_size[0]
                if args.debug and not _first_image_done:
                    print(f"[SW:LIT] window_size={model.window_size}")

                with torch.autocast(dtype=_ac_dtype, device_type=device.type):
                    crops_lit, origins_lit = model.window_imgs_semantic(imgs)
                    n_crops = len(crops_lit)

                    if logits_h is None:
                        logits_h, logits_w = H, W
                        print(f"[SW:LIT] first image: orig_hw={orig_hw} "
                              f"n_crops={n_crops} origins={origins_lit}")

                    # model(crops) — same rationale as branch C: go through
                    # the Lightning forward to get the /255 normalization.
                    mask_logits_layers, class_logits_layers = model(crops_lit)
                    mask_logits_lit = F.interpolate(
                        mask_logits_layers[-1], model.img_size, mode="bilinear"
                    )
                    crop_logits_lit = model.to_per_pixel_logits_semantic(
                        mask_logits_lit, class_logits_layers[-1]
                    )
                    logits_lit = model.revert_window_logits_semantic(
                        crop_logits_lit, origins_lit, img_sizes
                    )
                    pixel_logits = logits_lit[0].float()

                if args.debug and not _first_image_done:
                    _dbg_tensor("pixel_logits final (lightning_sw)", pixel_logits)
                    print(f"  [DBG:LIT] pixel_logits max={pixel_logits.max().item():.4f}")

                anomaly = _anomaly_from_pixel_logits(
                    pixel_logits,
                    method=args.method,
                    temperature=args.temperature,
                ).unsqueeze(0)

                anomaly_hw = tuple(anomaly.shape[-2:])
                gt_hw      = tuple(ood.shape[-2:])
                if anomaly_hw != gt_hw:
                    print(f"  [WARN:LIT_SHAPE] mismatch anomaly={anomaly_hw} gt={gt_hw} "
                          f"on {os.path.basename(p)}")
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

            # ── Branch A / A-interp: custom SlidingWindow ───────────────
            else:
                # The custom SlidingWindow utility produces square crops of
                # side min(H, W) directly from the original-resolution image
                # tensor. Crops are batched, forwarded through the model
                # (one micro-batch at a time), converted to per-pixel logits
                # via sigmoid(mask) @ softmax(class), and accumulated back
                # into the full image canvas. finalize() returns the
                # averaged per-pixel logit map.
                sw = SlidingWindow(img_size=min(H, W), device=device)
                crops, origins, _ = sw.window_image(x)
                n_crops = len(crops)

                if logits_h is None:
                    logits_h, logits_w = H, W
                    print(f"[SW] first image: orig_hw={orig_hw} n_crops={n_crops} "
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
                            _dbg_tensor("pixel_logits after to_pixel_logits (crop 0)", pl)

                        indices = list(range(crop_idx, crop_idx + batch.shape[0]))
                        sw.accumulate(pl.float(), origins, orig_hw, indices)
                        crop_idx += batch.shape[0]

                pixel_logits = sw.finalize(orig_hw)  # [C, H, W]

                if args.debug and not _first_image_done:
                    _dbg_tensor("pixel_logits final (after finalize, float32)", pixel_logits)
                    pl_max = pixel_logits.max().item()
                    pl_mean = pixel_logits.mean().item()
                    # Heuristic: float16 autocast typically compresses
                    # logits into a small range; if max is much larger than
                    # ~5 it usually means autocast was not effective.
                    if pl_max > 5.0:
                        print(f"  [DBG:AUTOCAST] pixel_logits max={pl_max:.4f} — "
                              f"unexpectedly high, autocast may not have been active")
                    else:
                        print(f"  [DBG:AUTOCAST] pixel_logits max={pl_max:.4f} mean={pl_mean:.4f} — "
                              f"compressed range, float16 autocast active ✓")

                anomaly = _anomaly_from_pixel_logits(
                    pixel_logits,
                    method=args.method,
                    temperature=args.temperature,
                ).unsqueeze(0)

                anomaly_hw = tuple(anomaly.shape[-2:])
                gt_hw      = tuple(ood.shape[-2:])
                if anomaly_hw != gt_hw:
                    print(f"  [WARN:SW_SHAPE] mismatch anomaly={anomaly_hw} gt={gt_hw} "
                          f"on {os.path.basename(p)} — bilinear upsample applied")
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

        # ── Per-image debug on the first processed image only ───────────
        if args.debug and not _first_image_done:
            _dbg_tensor("anomaly map (pre-squeeze)", anomaly)
            anomaly_np = anomaly.squeeze(0).detach().cpu().float().numpy()
            _dbg_anomaly_score_distribution(
                f"first image [{os.path.basename(p)}]",
                anomaly_np,
                ood
            )
            _first_image_done = True

        anomaly_list.append(anomaly.squeeze(0).detach().cpu().float().numpy())
        ood_list.append(ood)
        names.append(os.path.basename(p))
        n_processed += 1

    # ── Per-dataset summary ──────────────────────────────────────────────
    print(f"\n[SUMMARY] total images:        {len(paths)}")
    print(f"[SUMMARY] processed:           {n_processed}")
    print(f"[SUMMARY] skipped no GT:       {n_skipped_no_gt}")
    print(f"[SUMMARY] skipped no OOD px:   {n_skipped_no_ood}")
    print(f"[SUMMARY] mask_unique_before={sorted(unique_before_all)}")
    print(f"[SUMMARY] mask_unique_after= {sorted(unique_after_all)}")

    if n_processed == 0:
        raise RuntimeError(
            "No images were processed. "
            f"mask_unique_before={sorted(unique_before_all)} "
            f"mask_unique_after={sorted(unique_after_all)}"
        )

    ood_gts        = np.array(ood_list)
    anomaly_scores = np.array(anomaly_list)

    # Global score distribution diagnostics. If OOD < InD at this stage,
    # the anomaly score has the wrong sign for the chosen method.
    if args.debug:
        print(f"\n[DBG:GLOBAL] dataset-level score distribution:")
        _dbg_anomaly_score_distribution("GLOBAL", anomaly_scores.ravel(), ood_gts.ravel())
        print(f"[DBG:GLOBAL] ood_gts shape={ood_gts.shape}  "
              f"unique={np.unique(ood_gts).tolist()}")
        print(f"[DBG:GLOBAL] anomaly_scores shape={anomaly_scores.shape}  "
              f"nan={np.isnan(anomaly_scores).sum()}  "
              f"inf={np.isinf(anomaly_scores).sum()}")

    # Pixel-level evaluation: pool all valid (non-255) pixels across the
    # dataset, then compute AUPRC and FPR at 95% TPR.
    ood_out = anomaly_scores[ood_gts == 1]
    in_out  = anomaly_scores[ood_gts == 0]

    print(f"[METRICS] OOD pixels used: {len(ood_out):,}  InD pixels used: {len(in_out):,}  "
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
    print(f"FINAL RESULTS")
    print(f"  dataset={args.dataset_name}  method={args.method}  T={args.temperature}")
    print(f"  sw={args.sliding_window}  ckpt={ckpt_basename}  sha1={ckpt_sha1_8}")
    print(f"  AUPRC:    {metrics['auprc_pct']:.4f}%")
    print(f"  FPR@95:   {metrics['fpr95_pct']:.4f}%")
    print(f"  images: {n_processed}  (skip_gt={n_skipped_no_gt} skip_ood={n_skipped_no_ood})")
    print(f"  pixel OOD={len(ood_out):,}  InD={len(in_out):,}")
    print(f"  crop: {H}x{W}  logits_ref: {logits_h}x{logits_w}")
    print("=" * 60)

    # ── Persist metrics ──────────────────────────────────────────────────
    json_path = art.results / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {json_path}")

    csv_path = art.results / "metrics.csv"
    append_metrics_csv(csv_path, metrics)
    print(f"[SAVED] {csv_path}")

    # ── Optionally cache the raw model outputs for downstream sweeps ─────
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