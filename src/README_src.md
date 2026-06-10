# `src/`

Source code for the anomaly segmentation evaluation pipeline —
model wrappers, inference runners, offline temperature sweeps, metrics, and utilities.

---

## Directory Structure

```
src/
├── eval/
│   └── miou_cityscapes.py           ← mIoU evaluation on Cityscapes val (500 images)
├── models/
│   ├── eomt_wrapper.py              ← EoMT robust wrapper (checkpoint loading, inference)
│   └── README_models.md
├── runners/
│   ├── run_erfnet_eval.py           ← ERFNet anomaly inference (MSP, MaxLogit, MaxEntropy)
│   ├── run_eomt_eval.py             ← EoMT anomaly inference (MSP, MaxLogit, MaxEntropy, RbA)
│   ├── run_eomt_eval_v2.py          ← EoMT inference v2 (sliding window support)
│   ├── sweep_temp_from_cache.py     ← Offline temperature sweep — ERFNet cached logits
│   ├── sweep_temp_from_cache_eomt.py← Offline temperature sweep — EoMT cached logits
│   ├── sweep_temp_from_cache_sw.py  ← Offline temperature sweep — sliding window variant
│   └── README_runners.md
├── scripts/
│   ├── collect_results.py           ← Aggregate all metrics.json into a summary table
│   └── download_cityscapes.py       ← Download Cityscapes val split to Colab disk
└── utils/
    ├── artifacts.py                 ← Run directory management and artifact discovery
    ├── determinism.py               ← Reproducibility control (seeds, cuDNN, TF32)
    ├── eomt_post.py                 ← EoMT post-processing and anomaly scoring formulas
    ├── ood_dataset.py               ← GT mask loading and per-dataset label remapping
    ├── ood_metrics.py               ← FPR@95TPR and AUPRC computation
    ├── sliding_window.py            ← Sliding window inference for high-resolution inputs
    └── README_utils.md
```

---

## Module Reference

### `eval/miou_cityscapes.py` — Semantic mIoU evaluation

Evaluates EoMT on the Cityscapes validation set (500 images, 19 classes).
Supports three checkpoints with a consistent evaluation protocol:

| Mode | Checkpoint | `num_classes` | Strategy |
|------|-----------|--------------|----------|
| `cityscapes` | `eomt_cityscapes.bin` | 19 | Direct evaluation |
| `coco` | `eomt_coco.bin` | 133 | COCO → Cityscapes class remapping; unmapped pixels treated as void |
| `finetuned` | `eomt_finetuned.bin` | 19 | Direct evaluation |

The COCO → Cityscapes remapping maps only classes with a direct semantic equivalent.
Pixels whose predicted COCO class has no Cityscapes counterpart are excluded from the IoU
computation — following the standard zero-shot cross-dataset evaluation protocol.

```bash
python3 -m src.eval.miou_cityscapes \
  --images-dir  /content/cityscapes/leftImg8bit/val \
  --gt-dir      /content/cityscapes/gtFine/val \
  --ckpt        /path/to/eomt_cityscapes.bin \
  --config      eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --mode        cityscapes \
  --output-json /path/to/results/miou_cityscapes.json
```

---

### `models/eomt_wrapper.py` — EoMT model wrapper

Robust wrapper for **EoMT** (CVPR 2025), a ViT-based mask architecture with a DINOv2 backbone.

**Output format:**
```
mask_logits   [B, Q, H, W]   spatial occupancy of each of the Q=100 learned queries
class_logits  [B, Q, C+1]    class scores per query (C classes + no-object token)
```
Only the final decoder layer outputs are used for inference.

**Checkpoint loading (`num_classes` convention):**
EoMT adds `+1` internally when building `class_head` — use `num_classes=19` for
`eomt_cityscapes.bin` (which has `class_head=(20, 768)`). The single `missing=1`
at load time is `pos_embed` (shape mismatch between 1024×1024 training resolution
and 640×640 inference resolution), handled automatically via positional embedding
interpolation at runtime.

**Fuzzy weight loading** (`_load_weights_robust`): strips Lightning-style wrappers,
common key prefixes (`network.`, `model.`, `module.`, `eomt.`), and silently skips
shape-mismatched parameters — compatible with all checkpoint formats without
modifying original files.

```python
model = EoMTWrapper(img_size=(640, 640), num_classes=19, num_q=100,
                    num_blocks=3, backbone_name="vit_base_patch14_reg4_dinov2",
                    masked_attn_enabled=True)
model.load(ckpt_path, device, mode="robust")

mask_logits, class_logits = model.forward_masks_and_classes(x)
```

---

### `runners/run_erfnet_eval.py` — ERFNet inference

Evaluates ERFNet (`NUM_CLASSES=20`) on anomaly segmentation datasets at 1024×512.

**Anomaly scoring:**

| Method | Formula |
|--------|---------|
| `maxlogit` | `−max(logits)` — temperature-invariant |
| `msp` | `1 − max(softmax(logits/T))` |
| `maxentropy` | `−Σ p·log p`, `p = softmax(logits/T)` |

```bash
python3 -m src.runners.run_erfnet_eval \
  --input        "/path/to/dataset/images/*.*" \
  --weights      "trained_models/erfnet_pretrained.pth" \
  --dataset-name RA21 \
  --method       msp \
  --temperature  1.0 \
  --mode         robust \
  --seed         0 \
  --deterministic \
  --artifacts-dir artifacts \
  --save-logits
```

`--save-logits` caches raw logits as `float32 [N, 20, 512, 1024]` `.npy` files,
enabling offline temperature sweeps without re-running inference.

---

### `runners/run_eomt_eval.py` — EoMT inference

Evaluates EoMT on anomaly segmentation datasets at 640×640 (override with `--resize HxW`).
Logits are produced at 160×160 (4× downsampled) and upsampled to input resolution
via bilinear interpolation before metric computation.

**Methods:** `msp`, `maxentropy`, `maxlogit`, `rba` — all formulas in `src/utils/eomt_post.py`.

```bash
python3 -m src.runners.run_eomt_eval \
  --input        "/path/to/dataset/images/*.*" \
  --ckpt         "path/to/eomt_cityscapes.bin" \
  --config       "eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" \
  --dataset-name RA21 \
  --method       rba \
  --temperature  1.0 \
  --mode         robust \
  --seed         0 \
  --deterministic \
  --artifacts-dir artifacts \
  --save-logits
```

`--save-logits` caches:
- `mask_logits` as `float16 [N, Q, h, w]`
- `class_logits` as `float16 [N, Q, C+1]`

Float16 halves disk usage with negligible precision loss for anomaly scoring.

---

### `runners/sweep_temp_from_cache*.py` — Offline temperature sweeps

Recompute metrics at multiple temperatures from cached logits — no model forward pass needed.

| Script | Model | Logit files required |
|--------|-------|---------------------|
| `sweep_temp_from_cache.py` | ERFNet | `<dataset>__logits.npy` |
| `sweep_temp_from_cache_eomt.py` | EoMT | `<dataset>__mask_logits_f16.npy` + `<dataset>__class_logits_f16.npy` |
| `sweep_temp_from_cache_sw.py` | EoMT (sliding window) | same as EoMT |

```bash
# EoMT sweep example
python3 -m src.runners.sweep_temp_from_cache_eomt \
  --dataset-name  RA21 \
  --artifacts-dir artifacts \
  --use-latest \
  --method        msp \
  --mode          robust \
  --temperatures  0.5 0.75 1.0 1.1 1.25 1.5 2.0
```

`--use-latest` auto-resolves the most recent run with a valid logit cache on disk
(uses `resolve_latest_run_dir_filtered` with `require_logits=True`).

> **Note:** temperature scaling has no effect on MaxLogit by design —
> MaxLogit operates on pre-softmax logits which are scale-invariant.
> Identical sweep results for that method are expected, not a bug.

---

### `runners/` — Execution modes

| `--mode` | cuDNN | FPR95 impl | Weight loading | Use case |
|----------|-------|------------|----------------|----------|
| `robust` | deterministic | discrete cumsum sweep | fuzzy (shape-matched) | reproducible results |
| `prof-exact` | benchmark=True | sklearn ROC | strict | reproduce instructor baseline |

---

### `utils/eomt_post.py` — EoMT anomaly scoring formulas

Single source of truth for all EoMT post-processing. Used by both inference runners
and offline sweep scripts.

| Function | Method | Formula |
|----------|--------|---------|
| `pixel_probs_from_masks` | MSP / MaxEntropy | `pixel[c,h,w] = Σ_q softmax(class/T)[q,c] · sigmoid(mask)[q,h,w]`, renormalized |
| `anomaly_from_pixel_probs` | MSP, MaxEntropy | `1 − max(P)` or `−Σ P·log P` |
| `anomaly_maxlogit_from_masks` | MaxLogit | `pixel_logits[c,h,w] = Σ_q sigmoid(mask)[q,h,w] · class_logits[q,c]`; score = `1 − max_c(pixel_logits)` |
| `rba_from_masks` | RbA | reliability = `conf · area^0.5`; normality = `max_q(reliability · mask_prob)`; anomaly = `1 − normality` |

MaxLogit is implemented separately from MSP/MaxEntropy because it must operate on
raw pre-softmax logits — using `log(pixel_probs)` as a proxy is mathematically
incorrect and produces significantly worse results.

---

### `utils/ood_dataset.py` — GT mask loading

Unified ground-truth loading across all five benchmarks.
`remap_ood_mask` normalizes dataset-specific encodings to `{0=InD, 1=OOD, 255=ignore}`.

| Dataset | Raw encoding | Remapping |
|---------|-------------|-----------|
| RoadAnomaly21 / RoadAnomaly | label 2 = OOD | `2 → 1` |
| RoadObstacle21 | binary `{0,1}` | no-op |
| LostAndFound / FS_LostFound_full | legacy multi-class or binary | auto-detected via `_is_already_binary_ood_mask()` |
| Fishyscapes Static | binary `{0,1}` | no-op |
| Streethazard | class 14 = anomaly | `14→void, <20→0, void→1` |

The LostAndFound double-remap guard prevents label inversion when the export
is already in binary format.

---

### `utils/ood_metrics.py` — FPR@95TPR and AUPRC

| Function | Mode | Implementation |
|----------|------|---------------|
| `fpr_at_95_tpr_sweep` | `robust` | vectorized `cumsum` — deterministic, O(N log N) |
| `fpr_at_95_tpr_roc` | `roc` / `sklearn` | `sklearn.metrics.roc_curve` — matches some paper baselines |

AUPRC computed via `sklearn.metrics.average_precision_score` directly in runners.

---

### `utils/artifacts.py` — Experiment artifact management

Run directories follow the naming convention:
```
artifacts/<dataset>/<model>/<timestamp>__<method>__T<T>__<mode>__<hash8>/
```
The 8-character hash is derived from the full run config — identical configs
produce the same hash, making duplicate detection trivial.

`resolve_latest_run_dir_filtered` discovers the most recent run matching a given
`(dataset, model, method, mode)` combination. With `require_logits=True` it verifies
that the expected `.npy` cache files exist on disk before selecting a run.

---

### `utils/determinism.py` — Reproducibility

`apply_determinism(mode, seed, deterministic)` controls all non-determinism sources:
seeds Python / NumPy / PyTorch, disables TF32 on Ampere GPUs, and sets cuDNN
deterministic mode. Pass `--seed 0 --deterministic` to all runners for reproducible results.

---

### `scripts/collect_results.py` — Result aggregation

Walks the entire `artifacts/` tree, reads all `metrics.json` files, and prints
a structured summary table across datasets, models, and methods.

```bash
python3 -m src.scripts.collect_results --artifacts-dir /path/to/artifacts
```

### `scripts/download_cityscapes.py` — Cityscapes download

Downloads the Cityscapes validation split directly to Colab local disk.
Requires `CITYSCAPES_USER` and `CITYSCAPES_PASS` set as Colab secrets.
Output: `/content/cityscapes/leftImg8bit/val` and `/content/cityscapes/gtFine/val`.

---

## Supported Datasets

| Name | `--dataset-name` | Anomaly type |
|------|-----------------|-------------|
| SMIYC RoadAnomaly21 | `RA21` | Road obstacles & animals |
| SMIYC RoadObstacle21 | `RO21` | Small road surface obstacles |
| Fishyscapes Lost & Found | `LAF` | Lost objects on road |
| Fishyscapes Static | `fs_static` | Statically placed anomalies |
| Road Anomaly | `RA` | General road anomalies |

---

## Hardware

All runners support AMP and are tested on **NVIDIA A100 80GB**.
Minimum viable GPU is T4 16GB for single-dataset runs;
the full temperature sweep on LAF (ERFNet logits ≈ 4 GB) requires A100.
