# README: OOD Evaluation Pipeline (ERFNet + EoMT)

This repository implements a unified, reproducible and computationally optimized pipeline for pixel-wise Out-of-Distribution (OOD) detection in semantic segmentation.

### Supported architectures:
* **ERFNet** (CNN baseline)
* **EoMT** (Transformer / MaskFormer-style decoder)

The goal is to standardize preprocessing, metrics and evaluation logic across models while preserving compatibility with instructor-provided baselines.

---

## Objectives
* Standardize preprocessing and OOD metric computation across architectures.
* Reproduce baseline results from course scripts.
* Provide dual execution modes:
  * **robust** → stable, deterministic, modern evaluation.
  * **prof-exact** → reproduction of instructor-style behavior.
* Ensure mathematically correct Temperature Scaling via raw logit caching.
* Centralize metric computation (single source of truth for FPR@95TPR).

---

## Pipeline Architecture
The evaluation system is organized into four logical layers:

1.  **Model Wrappers**: Standardized loading logic handling prefix mismatches, DataParallel artifacts and checkpoint inconsistencies.
2.  **Post-Processing Layer**: Decouples model output from anomaly score computation (softmax, entropy, RBA, etc.).
3.  **Centralized OOD Metrics**: All FPR@95TPR and AUPRC logic implemented in `src/utils/ood_metrics.py`.
4.  **Execution Runners**: Inference runners + offline temperature sweep tools.

---

## Execution Modes (--mode)
Each runner supports:

### `mode = robust`
* FPR@95TPR computed via deterministic discrete threshold sweep.
* `cudnn.benchmark = False` for reproducibility.
* Flexible `state_dict` loading (prefix mismatches tolerated).
* Designed for stability, correctness and fair comparison across experiments.

### `mode = prof-exact`
* FPR@95TPR computed via sklearn ROC curve interpolation.
* `cudnn.benchmark = True` (matching course scripts).
* Strict `state_dict` loading.
* Designed to reproduce instructor baseline behavior.

---

## Dataset Handling
Centralized in `src/utils/ood_dataset.py`.

### Image → GT Mapping
Ground-truth masks are retrieved by replacing `images` with `labels_masks` in the file path. Extension remapping is handled automatically.

### Dataset-Specific Label Remapping
To obtain a consistent binary OOD mask:
* **RoadAnomaly**: 2 → 1 (OOD)
* **LostAndFound / FS_LostFound_full**: Complex remapping of void and class IDs to binary OOD.
* **Streethazard**: Specific remapping to isolate anomaly regions.

**Final convention:**
* **InD = 0**
* **OOD = 1**

Images without OOD pixels are automatically filtered to prevent metric distortion.

---

## Centralized Metrics
**File:** `src/utils/ood_metrics.py`

### AUPRC
Computed via `sklearn.metrics.average_precision_score`.
* **scores**: anomaly score per pixel.
* **labels**: binary OOD mask.

### FPR@95TPR
Two implementations:

#### 1. Sweep-based (robust)
Vectorized implementation:
* Sort scores descending.
* Compute cumulative TP and FP.
* Find first threshold where $TP \geq \lceil 0.95 \times P \rceil$.
* Return $FP / N$.

**Advantages:**
* Deterministic (no interpolation).
* Fast (NumPy C-ops, no Python loops).
* Scales to millions of pixels.

#### 2. ROC-based (prof-exact)
Uses `sklearn.metrics.roc_curve`. May differ slightly due to threshold interpolation.



---

## ERFNet Pipeline
### `run_erfnet_eval.py`
Primary inference runner.
* **Input resizing**: Bilinear | **GT resizing**: Nearest.
* **Supported anomaly methods**: MSP (1 − max softmax), MaxLogit, MaxEntropy.
* **Temperature Scaling**: Correct implementation via $Softmax(Logits / T)$.
* If `--save-logits` is enabled, raw logits $[N, C, H, W]$ are saved in `float32` to enable offline sweeps.

### `sweep_temp_from_cache.py`
Offline tool that loads cached logits and applies temperature scaling mathematically. Recomputes AUPRC + FPR instantly without requiring a GPU.

---

## EoMT Pipeline
### `eomt_wrapper.py`
Robust wrapper handling instructor checkpoint prefixes, MaskFormer-style outputs, and dynamic backbone configuration.

### `eomt_post.py`
Separates decoding from anomaly logic:
* **pixel_probs_from_masks**: Einsum composition of class and mask logits.
* **rba_from_masks**: Region-Based Anomaly (confidence × area reliability).

### `run_eomt_eval.py`
Mirrors ERFNet logic but adapted to transformer outputs. If `--save-logits` is enabled:
* Mask logits stored as `float16` (disk optimization).
* Class logits stored as `float16`.
* GT + names cached for sweep.

### `sweep_temp_from_cache_eomt.py`
Offline temperature sweep identical in philosophy to ERFNet.

---

## Computational Improvements
Compared to instructor scripts:
* **Fixed tensor dimension bug**: Corrected invalid `permute` after `ToTensor` in ERFNet evaluation.
* **Corrected anomaly score**: Used proper MSP instead of raw max logits.
* **Unified T-Scaling**: Applied to raw logits, not cached probabilities.
* **Vectorized FPR**: NumPy cumulative sums for speed.
* **Artifact Management**: Timestamped run folders for experiment traceability.

---

## Artifact System
Each run creates:
`artifacts/{dataset}/{model}/{timestamp__method__T__mode__hash}/`

**Contains:**
* `config.json`
* `results/metrics.json`
* `results/metrics.csv`
* `logits/` (optional)
* `sweep/` (temperature sweep outputs)

---

## Solved Issues
* Metric divergence across models.
* Inconsistent FPR@95 implementations.
* Incorrect anomaly score definitions in baseline scripts.
* Invalid temperature scaling and non-deterministic cuDNN behavior.
* Disk inefficiency during sweep experiments.

---

## Final Design Philosophy
The pipeline ensures mathematical correctness, reproducibility, and a controlled comparison between architectures through an efficient experimentation workflow compatible with instructor baselines.