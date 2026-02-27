# Anomaly Segmentation Project – Foundamental AI (PoliTo)

## Project Overview

This repository contains the structured and modular implementation of anomaly segmentation baselines for the Foundamental AI course.

The goal is to:
- Evaluate pixel-based segmentation (ERFNet) with post-hoc anomaly methods
- Evaluate mask-based architectures (EoMT)
- Compare anomaly scoring strategies
- Perform temperature scaling calibration
- Produce reproducible and modular experiments

We reorganized the original professor’s repository into a cleaner, research-oriented structure without modifying the baseline logic.

---

# Repository Structure

- **Foundamental_Ai_private/**
  - **src/**: Our evaluation pipeline (modular)
    - **runners/**: Modular scripts for experiments
    - **models/**: Clean model wrappers (ERFNet)
    - **utils/**: Metrics and helper utilities
  - **eval/**: Professor's original ERFNet baseline code
  - **eomt/**: Professor's mask-based architecture
  - **trained_models/**: Pretrained weights (if lightweight)
  - `README.md`
  - `requirements.txt`
  - `.gitignore`

---

# Professor Files (Untouched)

The following directories contain the original material provided:
- `eval/` -> ERFNet baseline implementation
- `eomt/` -> Mask-based architecture (EoMT)
- `trained_models/` -> Provided pretrained weights

We **do NOT modify** baseline logic. All experiments are run through our own modular runners in `src/runners/`.

---

# Google Drive Structure (External Storage)

Due to dataset and artifact size limitations, large files are stored on Google Drive.

## Dataset Location
Path: `/content/drive/MyDrive/anom_project/Validation_Dataset/`

Contains:
- RoadAnomaly21 (RA21)
- RoadObstacle21 (RO21)
- FS_LostFound_full
- fs_static
- RoadAnomaly

*Datasets are NOT versioned in GitHub.*

---

## Model Checkpoints
Path: `/content/drive/MyDrive/anomaly_project/ckpts/`

Pretrained or fine-tuned models are stored here to avoid pushing large `.pth` files to GitHub.

---

## Artifacts (Saved Outputs)
Path: `/content/drive/MyDrive/anom_project/artifacts/`

**Structure:**
- **artifacts/**
  - `logits/`: Cached logits (.npy)
  - `results/`: JSON results + `metrics.csv`

Artifacts are stored on Drive because logits are large, GitHub has file size limits, and we want reproducible temperature scaling without rerunning inference. Only lightweight JSON + CSV summaries are optionally pushed to GitHub.

---

# Experimental Pipeline

## 1. ERFNet Baseline (Pixel-Based)
**Runner:** `src/runners/run_erfnet_eval.py`

**Methods implemented:**
- MSP
- MaxLogit
- MaxEntropy

**Example:**
```bash
python -m src.runners.run_erfnet_eval \
  --input "/path/to/images/*.*" \
  --weights "trained_models/erfnet_pretrained.pth" \
  --method msp \
  --temperature 1.0 \
  --dataset-name RA21 \
  --artifacts-dir "/path/to/artifacts" \
  --save-logits

## 2. Temperature Scaling (From Cached Logits)
**Runner:** `src/runners/sweep_temp_from_cache.py`

**Example:**
```bash
python -m src.runners.sweep_temp_from_cache \
  --dataset-name RA21 \
  --artifacts-dir "/path/to/artifacts" \
  --temperatures "0.5,0.75,1.0,1.1,1.25,1.5,2.0,3.0"

  ---

## 2. Temperature Scaling (From Cached Logits)
**Runner:** `src/runners/sweep_temp_from_cache.py`

**Example:**
```bash
python -m src.runners.sweep_temp_from_cache \
  --dataset-name RA21 \
  --artifacts-dir "/path/to/artifacts" \
  --temperatures "0.5,0.75,1.0,1.1,1.25,1.5,2.0,3.0"

  ---

### Key Result
> "Temperature scaling has negligible impact on performance, suggesting that poor OOD performance is not due to miscalibration but to lack of separability in feature space."

---

### Why We Modularized the Code
The original professor code had hardcoded paths and mixed evaluation logic. We introduced:
- **Clean runners**: Explicit dataset naming.
- **Artifact separation**: Keeping heavy files off GitHub.
- **Logit caching**: Fast re-evaluation.

This ensures reproducibility and easy extension to newer models like EoMT.

---

### Current Status
- [x] ERFNet baseline implemented
- [x] MSP / MaxLogit / MaxEntropy
- [x] Logit caching system
- [x] Temperature scaling sweep
- [x] Drive-based artifact management
- [x] Clean GitHub repo (no heavy files)

---

### Next Steps
- Implement EoMT evaluation
- Add RbA scoring
- Evaluate all required datasets
- Produce final comparative results table