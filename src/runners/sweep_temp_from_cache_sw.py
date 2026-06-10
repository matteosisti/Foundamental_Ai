# src/runners/sweep_temp_from_cache_sw.py
#
# Offline temperature sweep for EoMT inference in sliding-window mode.
#
# This script reads the per-image pixel logits previously cached by
# run_eomt_eval.py when launched with --sliding-window --save-logits, and
# recomputes the anomaly metrics (AUPRC, FPR@95TPR) under a list of
# temperatures, without re-running the model.
#
# Cached files used (under <run_dir>/logits/):
#   * {dataset}__pixel_logits_f16.npy : [N, C, H_model, W_model] float16 —
#     per-image per-pixel logits at the MODEL resolution (e.g. 512x1024
#     after the SW preprocessing), NOT at the original image resolution.
#     Anomaly maps are bilinearly upsampled to the GT resolution before
#     scoring. The "pixel logits" here are the output of
#     sigmoid(mask) @ softmax(class), so they live in [0, ~1]; see
#     run_eomt_eval.py for the rationale.
#   * {dataset}__gt.npy               : [N, H, W] uint8 — ground truth
#     OOD masks (0 = in-distribution, 1 = OOD, 255 = ignore).
#
# Supported scoring methods:
#   * msp        : 1 - max(softmax(logits / T))
#   * maxlogit   : -max(logits)                      [temperature-invariant]
#   * maxentropy : -sum(p * log p), p = softmax(logits / T)
#   * rba        : -sum(tanh(logits), dim=C)         [temperature-invariant]
#
# The formulas are kept exactly aligned with _anomaly_from_pixel_logits in
# run_eomt_eval.py so that the T=1.0 point of the sweep matches the value
# the runner would have produced on the same cache.
#
# A typical workflow is:
#   1. Run the full inference once with --save-logits to populate the
#      cache.
#   2. Use this script to scan temperatures cheaply.

import os
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.utils.ood_metrics import fpr_at_95_tpr
from src.utils.determinism import apply_determinism
from src.utils.artifacts import resolve_latest_run_dir_filtered


def append_metrics_csv(csv_path: Path, row: dict) -> None:
    """Append one row to a metrics CSV, writing the header on first creation."""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def anomaly_from_pixel_logits_t(
    pixel_logits: torch.Tensor,
    method: str,
    T: float,
) -> torch.Tensor:
    """
    Compute an anomaly map from cached pixel logits at temperature T.

    Args:
        pixel_logits: tensor of shape [N, C, H, W] in float32. Comes from
                      the SW cache after the float16 -> float32 cast.
        method:       one of "msp", "maxlogit", "maxentropy", "rba".
        T:            temperature for MSP and MaxEntropy. Ignored by
                      MaxLogit and RbA (temperature-invariant by
                      construction).

    Returns:
        Tensor of shape [N, H, W] with per-pixel anomaly scores.
        Higher score = more anomalous.

    The formulas mirror exactly _anomaly_from_pixel_logits in
    run_eomt_eval.py so that the T=1.0 sweep point reproduces the runner's
    output bit by bit on the same cached tensors.
    """
    if method == "maxlogit":
        # Negate the max: a low max-logit means the model is uncertain
        # about the class, which we want to score as anomalous.
        return -pixel_logits.max(dim=1).values

    if method == "rba":
        # Reference RbA formula. Applied to pixel logits already in [0, 1]
        # (the SW pipeline output) it produces a compressed distribution,
        # which is a documented structural limitation in this mode.
        return -torch.tanh(pixel_logits).sum(dim=1)

    scaled = pixel_logits / T

    if method == "msp":
        probs = scaled.softmax(dim=1)
        return 1.0 - probs.max(dim=1).values

    if method == "maxentropy":
        probs = scaled.softmax(dim=1)
        return -(probs * (probs + 1e-8).log()).sum(dim=1)

    raise ValueError(f"Unknown method: {method}")


def main():
    ap = argparse.ArgumentParser(
        description="Offline temperature sweep from cached sliding-window pixel logits."
    )

    # ── Cache identification ─────────────────────────────────────────────
    ap.add_argument("--dataset-name",  required=True,
                    help="Dataset name used in the original run, e.g. "
                         "'RO21_sw_1024'. Used both to locate the cache "
                         "files and to name the metric prefix.")
    ap.add_argument("--artifacts-dir", default="artifacts",
                    help="Artifacts root. The script looks under "
                         "<artifacts-dir>/<dataset-name>/EoMT/...")
    ap.add_argument("--use-latest",    action="store_true",
                    help="Locate the most recent run directory that "
                         "matches dataset + method + mode and that has a "
                         "complete logits cache.")
    ap.add_argument("--run-dir",       default=None,
                    help="Explicit run directory. Alternative to --use-latest.")

    # ── Scoring configuration ────────────────────────────────────────────
    ap.add_argument("--method",        choices=["msp", "maxentropy", "maxlogit", "rba"],
                    default="msp",
                    help="Anomaly scoring method.")
    ap.add_argument("--mode",          choices=["robust", "prof-exact"], default="robust",
                    help="Metric mode for FPR@95TPR.")
    ap.add_argument("--temperatures",  default="0.5,0.75,1.0,1.1,1.25,1.5,2.0",
                    help="Comma-separated list of temperatures to scan.")

    # ── Runtime ──────────────────────────────────────────────────────────
    ap.add_argument("--device",        choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")

    args = ap.parse_args()

    apply_determinism(mode=args.mode, seed=args.seed, deterministic=args.deterministic)

    device = torch.device(
        "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    T_list = [float(x.strip()) for x in args.temperatures.split(",") if x.strip()]

    # ── Locate the run directory ─────────────────────────────────────────
    if args.run_dir is not None:
        run_root = Path(os.path.expanduser(args.run_dir))
    else:
        if not args.use_latest:
            raise ValueError("Provide --run-dir or pass --use-latest.")
        ds = args.dataset_name
        run_root = resolve_latest_run_dir_filtered(
            artifacts_root=args.artifacts_dir,
            dataset=ds,
            model="EoMT",
            method=args.method,
            mode=args.mode,
            require_logits=True,
            logit_files=[f"{ds}__pixel_logits_f16.npy", f"{ds}__gt.npy"],
        )

    print(f"[ARTIFACTS] {run_root}")

    # ── Load the cached tensors ──────────────────────────────────────────
    ds         = args.dataset_name
    logits_dir = run_root / "logits"
    pl_path    = logits_dir / f"{ds}__pixel_logits_f16.npy"
    gt_path    = logits_dir / f"{ds}__gt.npy"

    for p in [pl_path, gt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing cache file: {p}")

    # Memory-map both arrays so we can process them image by image without
    # ever holding the full dataset on the GPU. The full-dataset path was
    # tried first but produced OOMs on datasets such as LostAndFound where
    # the model-resolution logits (e.g. 512x1024) get upsampled to the GT
    # resolution (e.g. 1024x2048) and the dense [N, C, H, W] tensor in
    # float32 exceeds tens of gigabytes.
    pixel_logits_np = np.load(pl_path, mmap_mode="r")   # [N, C, H_model, W_model] float16
    gt              = np.load(gt_path)                  # [N, H_gt, W_gt] uint8

    N           = pixel_logits_np.shape[0]
    gt_h, gt_w  = int(gt.shape[-2]), int(gt.shape[-1])
    model_h, model_w = int(pixel_logits_np.shape[-2]), int(pixel_logits_np.shape[-1])
    need_upsample = (model_h, model_w) != (gt_h, gt_w)

    if need_upsample:
        print(f"[INFO] will upsample pixel_logits {(model_h, model_w)} "
              f"-> {(gt_h, gt_w)} per image (bilinear)")
    print(f"[INFO] N={N} images, processing one at a time on {device}")

    # ── Output directory ─────────────────────────────────────────────────
    sweep_dir = run_root / "sweep" / f"{args.method}__sw__{args.mode}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # ── Run the sweep ────────────────────────────────────────────────────
    # For every temperature we accumulate two flat 1D arrays in float32:
    # one with anomaly scores for InD pixels, one for OOD pixels. We never
    # need to keep full anomaly maps in memory, which makes this loop fit
    # in O(InD + OOD) RAM regardless of resolution.
    sweep_rows = []

    for Tv in T_list:
        ind_buf, ood_buf = [], []

        for i in range(N):
            # Load one image to GPU as float32. Shape [1, C, H_model, W_model].
            pl_i = torch.from_numpy(
                pixel_logits_np[i].astype(np.float32)
            ).unsqueeze(0).to(device)

            if need_upsample:
                pl_i = F.interpolate(
                    pl_i, size=(gt_h, gt_w),
                    mode="bilinear", align_corners=False,
                )

            anomaly_i = anomaly_from_pixel_logits_t(
                pl_i, method=args.method, T=Tv
            ).squeeze(0).cpu().numpy()  # [H_gt, W_gt]

            gt_i = gt[i]
            ind_buf.append(anomaly_i[gt_i == 0])
            ood_buf.append(anomaly_i[gt_i == 1])

            del pl_i, anomaly_i

        in_out  = np.concatenate(ind_buf) if ind_buf else np.empty(0)
        ood_out = np.concatenate(ood_buf) if ood_buf else np.empty(0)

        val_out   = np.concatenate([in_out, ood_out])
        val_label = np.concatenate([np.zeros(len(in_out)), np.ones(len(ood_out))])

        auprc = float(average_precision_score(val_label, val_out))
        fpr95 = float(fpr_at_95_tpr(val_out, val_label, mode=args.mode))

        row = {
            "T":         Tv,
            "method":    args.method,
            "mode":      args.mode,
            "sw":        True,
            "auprc":     auprc,
            "fpr95":     fpr95,
            "auprc_pct": auprc * 100.0,
            "fpr95_pct": fpr95 * 100.0,
        }

        t_str    = str(Tv).replace(".", "p")
        out_path = sweep_dir / f"T{t_str}__metrics.json"
        with open(out_path, "w") as f:
            json.dump(row, f, indent=2)

        sweep_rows.append(row)
        print(f"[T={Tv}] AUPRC={auprc*100:.4f} | FPR95={fpr95*100:.4f} | saved {out_path}")

    # ── Aggregate CSV ────────────────────────────────────────────────────
    csv_path = sweep_dir / "metrics_sweep.csv"
    for row in sweep_rows:
        append_metrics_csv(csv_path, row)

    print(f"[DONE] Sweep results in: {sweep_dir}")


if __name__ == "__main__":
    main()