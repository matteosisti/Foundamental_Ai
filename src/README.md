# README: OOD Evaluation Pipeline (ERFNet + EoMT)

This repository implements a unified and reproducible pipeline for Out-of-Distribution (OOD) detection in semantic segmentation, supporting both Convolutional (ERFNet) and Transformer-based (EoMT) architectures.

---

## Objectives
* Standardize preprocessing and metrics across different architectures.
* Reproduce baseline results provided in the course scripts.
* Support Dual Modes: robust (standard/stable) and prof-exact (instructor-specific).
* Enable Correct Temperature Scaling via raw logit caching.
* Centralize Metrics: Single source of truth for FPR@95.

---

## Architecture
The pipeline is structured into four functional layers:

1. Model Wrappers: Standardized loading and interface (handling module prefixes).
2. Post-processing: Decoupled logic for probability and anomaly score calculation.
3. Centralized OOD Metrics: Unified implementation of AUPRC and FPR@95.
4. Execution Runners: Inference engines and offline temperature sweep tools.

---

## Execution Modes (--mode)
Each runner supports a mode flag to ensure experimental consistency:

### mode = robust
* Metric: FPR@95 calculated via an explicit threshold sweep.
* Environment: cudnn.benchmark disabled for maximum determinism.
* Loading: Flexible weight loading (handles prefix mismatches).
* Focus: Code stability and modern evaluation standards.

### mode = prof-exact
* Metric: FPR@95 calculated via Scikit-Learn ROC curve (matching instructor scripts).
* Environment: cudnn.benchmark enabled (matching course baselines).
* Loading: Strict state_dict loading.
* Focus: Precise reproduction of provided benchmarks.

---

## Core Components

### 1. Centralized Metrics (src/utils/ood_metrics.py)
Eliminates discrepancies between different model implementations.
* fpr_at_95_tpr_sweep: Manual calculation by sorting scores (Robust).
* fpr_at_95_tpr_roc: Calculation via sklearn.metrics.roc_curve (Prof-exact).

### 2. Dataset Management (src/utils/ood_dataset.py)
Centralizes dataset-specific logic previously scattered across scripts.
* Mapping: Image-to-GT path synchronization.
* Remapping: Specific label handling for RoadAnomaly, LostAndFound, and StreetHazards.
* Validation: Automatically filters images without OOD pixels to prevent metric distortion.

---

## ERFNet Pipeline

### run_erfnet_eval.py
The primary inference runner. It handles input resizing (Bilinear) and GT resizing (Nearest).
* Anomaly Methods: Supports MSP, MaxLogit, and MaxEntropy.
* Caching: If --save-logits is active, it saves raw logits [N, C, H, W] in float32.

### sweep_temp_from_cache.py
An offline tool for Temperature (T) optimization.
* Why it matters: Previous versions cached probabilities. By caching raw logits, we can mathematically apply Softmax(Logits/T) correctly during the sweep, ensuring valid Temperature Scaling.

---

## EoMT Pipeline

### eomt_wrapper.py
A robust wrapper to handle instructor checkpoints. It ensures the MaskFormer-style decoder outputs are correctly formatted regardless of the training prefix.

### eomt_post.py
Separates model output from anomaly score calculation:
* pixel_probs_from_masks: Composes class and mask logits via einsum.
* rba_from_masks: Implements Region-Based Anomaly detection (Confidence, Area, and Reliability).

### run_eomt_eval.py and sweep_temp_from_cache_eomt.py
Mirror the ERFNet logic but tailored for Transformer outputs. Caching uses float16 for mask logits to balance disk space and numerical precision.

---

## Solved Issues
* Metric Divergence: Unified FPR@95 across all models.
* Mathematical Correctness: Temperature scaling is now applied to raw logits.
* Reproducibility: The --mode flag allows for apples-to-apples comparisons.
* Efficiency: Caching allows for near-instant temperature optimization without re-running GPU inference.