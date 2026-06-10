# Comprehensive Road Scene Understanding for Autonomous Driving

> **Course project** — Politecnico di Torino  
> TAs: Alessandro Marinai · Davide Sferrazza · Stephany Ortuno

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20A100-orange?logo=googlecolab)](https://colab.research.google.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20A100%2080GB-76b900?logo=nvidia)](https://www.nvidia.com/en-us/data-center/a100/)

---

## Overview

This project tackles **comprehensive road scene understanding** for autonomous driving, spanning three progressively challenging tasks:

1. **Semantic Segmentation** — per-pixel class labelling with ERFNet and EoMT
2. **Panoptic Segmentation** — unified stuff + things understanding with EoMT
3. **Anomaly Segmentation** — out-of-distribution object detection with post-hoc scoring methods

The pipeline covers the full lifecycle: model study, quantitative evaluation on Cityscapes, fine-tuning from a COCO checkpoint, and anomaly benchmarking across five real-world road datasets.

---

## Repository Structure

```
.
├── notebooks/
│   ├── README_notebooks.md              ← Detailed notebook documentation
│   ├── anomaly_eval_pipeline.ipynb      ← End-to-end anomaly evaluation (ERFNet + EoMT)
│   ├── step4_miou_eval.ipynb            ← Step 4: semantic mIoU comparison (COCO vs CS)
│   ├── EDA/
│   │   ├── EDA.ipynb                    ← Exploratory data analysis on Cityscapes
│   │   └── eda_from_results.ipynb       ← Result-level EDA & visualisation
│   ├── step5/
│   │   ├── README.MD                    ← Fine-tuning details & result table
│   │   ├── miou_full_evaluation_pipeline.ipynb
│   │   ├── step5_sweep.ipynb            ← Full FT: bs=32, 40 ep → 64.03% mIoU
│   │   ├── step5_BS2.ipynb              ← Full FT: bs=2 baseline
│   │   ├── step5_bs32_epoch110.ipynb    ← Extended FT run
│   │   ├── step5_lora_on_sweep.ipynb    ← LoRA on top of sweep ckpt → 64.24% mIoU
│   │   └── result_collector/
│   │       ├── step8_anomaly_finetuned_lora_on_sweep.ipynb
│   │       └── step8_anomaly_finetuned_sweep_bs32_epoch50.ipynb
│   └── step8/
│       ├── Step8_COCO.ipynb             ← Anomaly eval — EoMT COCO checkpoint
│       ├── Step8_COCO_SWEEP.ipynb       ← Anomaly eval — COCO + temperature sweep
│       ├── stage8_COCO_RA.ipynb         ← COCO checkpoint on Road Anomaly dataset
│       ├── step8_CS.ipynb               ← Anomaly eval — EoMT Cityscapes checkpoint
│       └── step8_FT.ipynb               ← Anomaly eval — fine-tuned checkpoint
└── README.md                            ← This file
```

---

## Models

| Model | Backbone | Training data | Task | Checkpoint |
|-------|----------|--------------|------|------------|
| ERFNet | Residual factorised ConvNet | Cityscapes | Semantic segmentation | GitHub (provided) |
| EoMT-CS | ViT-L + DINOv2 | Cityscapes | Semantic segmentation | Drive `ckpts/eomt_cityscapes.bin` |
| EoMT-COCO | ViT-L + DINOv2 | COCO | Panoptic segmentation | Drive `ckpts/eomt_coco.bin` |
| EoMT-FT (sweep) | ViT-L + DINOv2 | COCO → Cityscapes FT | Semantic segmentation | Drive `sweep_bs48/` — **64.03% mIoU** |
| EoMT-FT (LoRA) | ViT-L + DINOv2 | sweep ckpt + LoRA | Semantic segmentation | Drive `lora_on_sweep/` — **64.24% mIoU** |

---

## Anomaly Segmentation Methods

### Post-hoc scoring (pixel-based — ERFNet)

| Method | Description |
|--------|-------------|
| MSP | Maximum Softmax Probability — `1 − max p(y|x)` |
| MaxLogit | Maximum pre-softmax logit score |
| Max Entropy | Entropy of the predicted distribution |

### Post-hoc scoring (mask-based — EoMT)

All pixel-based methods above, plus:

| Method | Description |
|--------|-------------|
| RbA | *Rejected by All* — scores pixels whose logits are rejected by every mask query |
| Temperature Scaling | Calibrated softmax with sweep over T ∈ {0.5, 0.75, 1.0, 1.1, …} |

> **Note:** temperature scaling has no effect on MaxLogit by design — MaxLogit is scale-invariant with respect to temperature. Identical sweep results for that method are expected behavior.

---

## Evaluation Benchmarks

| Dataset | Split | Images | Anomaly type |
|---------|-------|--------|-------------|
| SMIYC RoadAnomaly21 (RA-21) | Val | 100 | Road obstacles & animals |
| SMIYC RoadObstacle21 (RO-21) | Val | 50 | Small obstacles |
| Fishyscapes Lost & Found (FS L&F) | Val | 100 | Lost objects on road |
| Fishyscapes Static (FS Static) | Val | 30 | Statically placed anomalies |
| Road Anomaly (RA) | Val | 60 | General road anomalies |

Metrics: **AuPRC** (primary) and **FPR95**.

---

## Fine-tuning Results (Step 5)

| Method | Epochs | Batch size | Trainable params | mIoU (Cityscapes val) |
|--------|--------|-----------|-----------------|----------------------|
| COCO zero-shot | — | — | — | 12.49% |
| Full FT (bs=2) — Stefano | 50 | 2 | 93.5M | 62.72% |
| Full FT (bs=2) — Beybin | 50 | 2 | 93.5M | 61.71% |
| **Full FT (sweep, bs=32)** | **40** | **32** | **93.5M** | **64.03%** |
| **LoRA on sweep (r=16)** | **50** | **32** | **~2M** | **64.24%** |
| CS baseline (pretrained) | — | — | — | 71.69% |

Training config (sweep): LR=1e-4, LLRD=0.9, warmup_steps=[92, 184], MaskFormer loss (BCE + Dice + CE + Hungarian matching), AMP enabled.

LoRA config: r=16, targets=qkv+proj layers, LR=5e-5.

---

## Hardware & Environment

| Resource | Spec |
|----------|------|
| **GPU** | NVIDIA A100 80GB (recommended) |
| Minimum GPU | T4 16GB (small datasets only — may OOM on LAF temperature sweep) |
| Platform | Google Colab Pro / Pro+ |
| CUDA | 12.x |
| Python | 3.10+ |
| PyTorch | 2.x |
| Mixed precision | AMP (torch.cuda.amp) — enabled for all training and inference |

> The temperature sweep loads all cached logits into VRAM simultaneously. ERFNet LAF logits alone are ~4 GB; running the sweep on T4 for that dataset will OOM. **A100 is strongly recommended** for the full pipeline.

---

## Google Drive Layout

The notebooks expect the following structure under `MyDrive/`:

```
MyDrive/
├── anom_project/                        ← shared evaluation folder
│   ├── ckpts/
│   │   └── eomt/
│   │       ├── eomt_cityscapes.bin
│   │       └── eomt_coco.bin
│   ├── Validation_Dataset/
│   │   ├── RoadAnomaly21/images/
│   │   ├── RoadObstacle21/images/
│   │   ├── LostAndFound/images/
│   │   ├── fs_static/images/
│   │   └── RoadAnomaly/images/
│   └── artifacts/                       ← auto-created by runners
│       └── <DATASET>/<MODEL>/
│           └── <timestamp>__<method>__T<T>__<mode>__<hash8>/
│               ├── config.json
│               ├── results/metrics.json
│               ├── logits/              ← cached .npy files
│               └── sweep/
└── anom_project_private/                ← fine-tuning outputs
    ├── sweep_bs48/
    └── lora_on_sweep/
```

---

## Quick Start

### 1. Set Colab Secrets

Before running any notebook, add the following in the Colab **Secrets** panel (🔑 left sidebar):

| Secret | Value |
|--------|-------|
| `GITHUB_USERNAME` | your GitHub username |
| `GITHUB_TOKEN` | personal access token (repo scope) |
| `GITHUB_REPO` | repository name |
| `USER_EMAIL` | your git email |

Credentials are read at runtime via `google.colab.userdata` and never appear in notebook output or git history.

### 2. Set Runtime

```
Runtime → Change runtime type → A100 GPU
```

### 3. Mount Drive & Clone

Each notebook handles Drive mounting and repo cloning in its first cell.

### 4. Run Evaluation

```
anomaly_eval_pipeline.ipynb   ← full ERFNet + EoMT anomaly benchmark
step4_miou_eval.ipynb         ← semantic mIoU (COCO vs CS checkpoint)
step5/step5_sweep.ipynb       ← fine-tuning from COCO
step8/Step8_COCO.ipynb        ← anomaly eval on fine-tuned checkpoints
```

---

## Reproducibility

- Random seed: `0` (fixed across all training runs)
- All checkpoints stored on Drive with SHA1 documented in `results/*.json`
- WandB project: `anom-ft-private`
- Git history tracks all config changes — clone from `main`, not ZIP

---

## References

1. [SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation](https://segmentmeifyoucan.com/)
2. [The Fishyscapes Benchmark](https://fishyscapes.com/)
3. [MaskFormer — Per-Pixel Classification is Not All You Need](https://arxiv.org/abs/2107.06278)
4. [Mask2Former](https://arxiv.org/abs/2112.01527)
5. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
6. [EoMT — Your ViT is Secretly an Image Segmentation Model (CVPR 2025)](https://arxiv.org/abs/2503.19108)
7. [RbA: Segmenting Unknown Regions Rejected by All](https://arxiv.org/abs/2211.14293)
8. [Scaling Out-of-Distribution Detection for Real-World Settings](https://arxiv.org/abs/2107.09751)
9. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
10. [ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/document/8063438)
11. [Panoptic Segmentation — Kirillov et al.](https://arxiv.org/abs/1801.00868)
12. [The Cityscapes Dataset](https://www.cityscapes-dataset.com/)
13. [Microsoft COCO](https://cocodataset.org/)
