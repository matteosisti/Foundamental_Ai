# EoMT COCO — Anomaly Segmentation Results

**Date**: May 08, 2026
**Checkpoint**: `eomt_coco.bin` — EoMT Base 640, trained on COCO panoptic
**Notebook**: `notebooks/step8_anomaly_coco.ipynb`
**Artifacts**: `artifacts/<dataset>_coco/EoMT/`
**Sweeps**: `results/all_sweeps_coco.json`

---

## Best results after temperature sweep

| Dataset | Best method | Best AuPRC | Best T | EoMT CS baseline |
|---------|------------|-----------|--------|-----------------|
| RA21 | MaxEntropy | 35.81% | 2.0 | 68.22% |
| RO21 | MaxEntropy | 1.07% | 0.5 | 91.52% |
| LAF | MaxEntropy | 0.43% | 2.0 | 22.41% |
| fs_static | MaxEntropy | 3.76% | 2.0 | 58.16% |
| RA | MaxEntropy | 17.94% | 2.0 | 73.08% |

---

## Full sweep results — AuPRC (↑)

### RA21
| Method | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|--------|-------|--------|-------|-------|--------|-------|-------|
| MSP | 15.15 | 14.87 | 15.18 | 15.51 | 16.14 | 17.57 | **22.17** |
| MaxEntropy | 15.37 | 14.61 | 15.96 | 17.46 | 20.11 | 25.36 | **35.81** |
| MaxLogit | 16.17 | 16.17 | 16.17 | 16.17 | 16.17 | 16.17 | 16.17 |
| RbA | 15.55 | 15.53 | 15.50 | 15.51 | 15.53 | 15.62 | 15.85 |

| Method (FPR95↓) | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|----------------|-------|--------|-------|-------|--------|-------|-------|
| MSP | 95.17 | 93.84 | 92.97 | 92.62 | 90.91 | 85.22 | **83.15** |
| MaxEntropy | 95.12 | 93.77 | 92.73 | 91.63 | 87.47 | **79.93** | 80.26 |
| MaxLogit | **66.56** | 66.56 | 66.56 | 66.56 | 66.56 | 66.56 | 66.56 |
| RbA | **65.25** | 66.84 | 67.18 | 67.55 | 67.70 | 67.85 | 67.33 |

### RO21
| Method | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|--------|-------|--------|-------|-------|--------|-------|-------|
| MSP | **0.89** | 0.81 | 0.73 | 0.70 | 0.67 | 0.64 | 0.67 |
| MaxEntropy | **1.07** | 1.02 | 0.92 | 0.86 | 0.85 | 0.87 | 0.99 |
| MaxLogit | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 |
| RbA | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 |

| Method (FPR95↓) | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|----------------|-------|--------|-------|-------|--------|-------|-------|
| MSP | **55.55** | 74.94 | 80.87 | 86.81 | 88.10 | 87.94 | 79.92 |
| MaxEntropy | **54.35** | 72.27 | 74.05 | 75.82 | 75.83 | 73.05 | 57.40 |
| MaxLogit | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| RbA | 75.71 | 75.74 | 75.77 | 75.79 | 75.81 | 75.81 | **75.70** |

### LostAndFound
| Method | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|--------|-------|--------|-------|-------|--------|-------|-------|
| MSP | 0.20 | 0.19 | 0.19 | 0.20 | 0.20 | 0.22 | **0.26** |
| MaxEntropy | 0.20 | 0.20 | 0.23 | 0.25 | 0.28 | 0.33 | **0.43** |
| MaxLogit | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 | 0.18 |
| RbA | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | **0.20** |

| Method (FPR95↓) | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|----------------|-------|--------|-------|-------|--------|-------|-------|
| MSP | 98.98 | 98.60 | 98.10 | 97.28 | 96.43 | 95.48 | **94.34** |
| MaxEntropy | 98.95 | 98.28 | 96.49 | 95.89 | 93.94 | 90.79 | **83.78** |
| MaxLogit | 94.29 | 94.29 | 94.29 | 94.29 | 94.29 | 94.29 | 94.29 |
| RbA | **86.37** | 88.38 | 89.34 | 89.67 | 89.93 | 90.08 | 90.22 |

### Fishyscapes Static
| Method | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|--------|-------|--------|-------|-------|--------|-------|-------|
| MSP | 2.03 | 2.12 | 2.24 | 2.32 | 2.44 | 2.65 | **3.16** |
| MaxEntropy | 2.41 | 2.55 | 2.89 | 3.14 | 3.34 | 3.50 | **3.76** |
| MaxLogit | 1.69 | 1.69 | 1.69 | 1.69 | 1.69 | 1.69 | 1.69 |
| RbA | 1.69 | 1.69 | 1.69 | 1.69 | 1.70 | 1.70 | **1.72** |

| Method (FPR95↓) | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|----------------|-------|--------|-------|-------|--------|-------|-------|
| MSP | 97.91 | 96.73 | 95.37 | 94.61 | 93.76 | 93.00 | **92.19** |
| MaxEntropy | 97.81 | 96.60 | 94.90 | 93.83 | 92.97 | 89.29 | **86.31** |
| MaxLogit | 93.26 | 93.26 | 93.26 | 93.26 | 93.26 | 93.26 | 93.26 |
| RbA | **84.20** | 85.55 | 86.57 | 86.97 | 87.67 | 88.06 | 87.48 |

### Road Anomaly
| Method | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|--------|-------|--------|-------|-------|--------|-------|-------|
| MSP | 8.89 | 8.69 | 8.63 | 8.67 | 8.82 | 9.25 | **10.53** |
| MaxEntropy | 8.98 | 8.78 | 9.60 | 10.25 | 11.52 | 13.55 | **17.94** |
| MaxLogit | 8.58 | 8.58 | 8.58 | 8.58 | 8.58 | 8.58 | 8.58 |
| RbA | 8.92 | 8.88 | 8.86 | 8.86 | 8.87 | 8.91 | **9.02** |

| Method (FPR95↓) | T=0.5 | T=0.75 | T=1.0 | T=1.1 | T=1.25 | T=1.5 | T=2.0 |
|----------------|-------|--------|-------|-------|--------|-------|-------|
| MSP | 94.20 | 94.23 | **93.44** | 93.22 | 93.39 | 95.16 | 96.71 |
| MaxEntropy | 94.16 | 93.67 | **92.36** | 92.55 | 93.96 | 94.32 | 94.24 |
| MaxLogit | 95.51 | 95.51 | 95.51 | 95.51 | 95.51 | 95.51 | 95.51 |
| RbA | **81.83** | 84.29 | 85.83 | 86.34 | 87.18 | 86.95 | 86.61 |

---

## Analysis

**RA21 is the strongest result** (MaxEntropy 35.81% at T=2.0) — the COCO model retains
some capacity to detect large anomalous objects, though significantly below the
Cityscapes baseline (68.22%). MaxEntropy monotonically improves with temperature,
suggesting the model is systematically over-confident on road scenes.

**RO21 and LAF collapse to near-zero** (MaxEntropy 1.07% and 0.43%) — both datasets
contain small obstacles on road surfaces. The model has no prior for road-context
anomalies. Notably on RO21, lower temperatures (T=0.5) give better results — the
opposite of RA21 — suggesting a different failure mode for small obstacle detection.

**MaxLogit is temperature-invariant** (confirmed on all datasets) and surprisingly
competitive on RA21 FPR95 (66.56%) despite low AuPRC — it achieves better threshold
separation than MSP/MaxEntropy on that dataset.

**RbA FPR95 is consistently better than AuPRC rank** — on RA (81.83% FPR95 at T=0.5)
and LAF (86.37% FPR95) RbA has the best FPR95 even though its AuPRC is not the highest.
This suggests RbA's mask-level scoring is better calibrated for threshold-based detection
even on a domain-mismatched checkpoint.

**Temperature scaling pattern** — MaxEntropy with T=2.0 gives best AuPRC on 4/5 datasets.
For FPR95 the optimal T is lower (T=0.5–1.0 on RO21, RA). The two metrics optimize
at different temperatures, confirming that AuPRC and FPR95 capture different aspects
of the anomaly detection performance.

**Overall conclusion** — the COCO checkpoint performs significantly worse than EoMT
Cityscapes across all datasets (delta: −32.41% on RA21, −90.45% on RO21). The gap
is most severe on small-obstacle datasets where road context is critical. This provides
strong quantitative motivation for fine-tuning on Cityscapes (Step 5).
