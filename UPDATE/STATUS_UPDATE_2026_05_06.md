# Project Status Update — May 06, 2026

## Overview

This document summarizes the current state of the project, what has been completed, what remains to be done, and realistic time estimates for each remaining task.

---

## Completed Work

**ERFNet pixel-based baselines (Project Step 7)**
All three post-hoc methods — MSP, MaxEntropy, and MaxLogit — have been evaluated on all five anomaly segmentation datasets: RoadAnomaly21, RoadObstacle21, LostAndFound, Fishyscapes Static, and Road Anomaly. Results are fully reproducible and stored in the artifacts directory with structured JSON output.

**EoMT mask-based baselines at 640x640 (Project Step 8 — partial)**
The Cityscapes-trained EoMT Base checkpoint has been evaluated on all five datasets with all four post-hoc methods: MSP, MaxEntropy, MaxLogit, and RbA. Temperature scaling sweep has been completed across seven temperature values (0.5 to 2.0) for all methods and datasets, with logit caching to avoid redundant forward passes.

**EoMT Large 1024x1024 ablation**
An ablation study at higher resolution has been conducted using the EoMT Large checkpoint on all five datasets with all four post-hoc methods. Results show significant variation across methods and datasets compared to the 640x640 baseline, with notable degradation in RbA and MaxEntropy that is attributed to aspect ratio distortion from square resizing.

**Infrastructure and tooling**
A full evaluation pipeline has been implemented from scratch, including structured artifact management with SHA1-based run identification, offline temperature sweep from cached logits, deterministic execution mode, and automated result collection across all runs. Everything is documented and version-controlled on GitHub.

---

## Remaining Work and Time Estimates

**Cityscapes mIoU evaluation — 4 to 6 hours**
Evaluation of EoMT Cityscapes and EoMT COCO checkpoints on the Cityscapes validation set (500 images, 19 classes). The evaluation script has been written and is ready to run. Download of the Cityscapes validation split is in progress. This also includes qualitative visualizations of semantic predictions (Cityscapes model) and panoptic predictions (COCO model) as required by Project Step 4.

**GT mask validation and aspect ratio fix — 2 to 3 hours**
The current evaluation pipeline resizes ground truth masks to a fixed square resolution, which distorts the road area evaluation region in RO21 and potentially other datasets. An aspect-ratio-preserving resize with padding needs to be implemented and the impact on the Large 1024 results needs to be quantified. This is also needed to validate that the anomaly maps are being evaluated correctly.

**Anomaly map qualitative analysis — 3 to 4 hours**
Heatmap visualizations of anomaly scores have not yet been generated. Visual inspection is needed to confirm that high-scoring regions correspond to actual anomalous objects and not to artifacts such as image borders or compression noise. A Colab cell to save heatmaps as PNG for a representative subset of images per dataset needs to be written.

**EoMT COCO anomaly segmentation (Project Step 8 — completion) — 4 to 6 hours**
The COCO-trained checkpoint has not yet been evaluated on the anomaly segmentation datasets. The existing runner supports this directly via checkpoint argument. Results need to be collected and compared against the Cityscapes checkpoint baseline.

**Fine-tuning EoMT COCO on Cityscapes (Project Step 5) — 30 to 40 hours**
This is the largest remaining task. It requires implementing a training loop with automatic mixed precision (AMP), checkpoint saving, loss monitoring, and convergence validation. The number of epochs needed for convergence is unknown and depends on runtime experiments. LoRA may be required if memory constraints on Colab prevent full fine-tuning. Once the fine-tuned checkpoint is available, mIoU evaluation and anomaly segmentation with the fine-tuned model need to be run.

**EoMT fine-tuned anomaly segmentation — 4 to 6 hours**
Once the fine-tuned checkpoint is available, anomaly segmentation evaluation follows directly using the existing runner. Temperature scaling sweep on the new checkpoint is also included in this estimate.

**Report — 15 to 20 hours**
A 5-page technical report covering all project steps. The results tables for anomaly segmentation are largely ready. Remaining work includes writing the Introduction, Related Work, Methods, and Results sections, generating figures, and integrating all quantitative and qualitative results into a coherent narrative.

---

## Total Estimate

| Task | Estimated Hours |
|------|----------------|
| Cityscapes mIoU evaluation and visualizations | 4 to 6 |
| GT mask validation and aspect ratio fix | 2 to 3 |
| Anomaly map qualitative analysis | 3 to 4 |
| EoMT COCO anomaly segmentation | 4 to 6 |
| Fine-tuning EoMT COCO on Cityscapes | 30 to 40 |
| Fine-tuned anomaly segmentation and sweep | 4 to 6 |
| Report | 15 to 20 |
| **Total** | **62 to 85 hours** |

At a sustained pace of 6 to 8 hours per day, the remaining work requires approximately 8 to 14 working days. The submission deadline is June 10, 2026, leaving sufficient time if work begins immediately. The critical path is the fine-tuning task, which is the most time-consuming and the most uncertain in terms of convergence behavior.

---

## Priority Order

1. Cityscapes download and mIoU evaluation — unblocks Step 4 immediately
2. EoMT COCO anomaly segmentation — uses existing infrastructure, fast to run
3. Anomaly map qualitative analysis — needed for report figures
4. GT mask aspect ratio fix — validates Large 1024 results
5. Fine-tuning — longest task, start as early as possible
6. Fine-tuned anomaly segmentation — depends on step 5
7. Report — written in parallel where possible

---

## Detailed Task Breakdown

### Block 1 — Cityscapes mIoU Evaluation (Step 4)

| # | Task | Description | Estimate |
|---|------|-------------|----------|
| 1.1 | Download Cityscapes val | Run download_cityscapes.py via Colab secrets, extract val split only (~1.8 GB) | 30 min |
| 1.2 | mIoU EoMT Cityscapes | Forward pass on 500 val images, compute per-class IoU and mIoU, save JSON | 1 hr runtime |
| 1.3 | mIoU EoMT COCO | Same pipeline with COCO->Cityscapes class remapping applied | 1 hr runtime |
| 1.4 | Semantic visualization | Save predicted segmentation overlay for 3-5 sample images using Cityscapes checkpoint | 1 hr coding |
| 1.5 | Panoptic visualization | Save panoptic prediction overlay for 3-5 sample images using COCO checkpoint | 1-2 hr coding |
| **Block 1 total** | | | **4-6 hrs** |

### Block 2 — Result Validation

| # | Task | Description | Estimate |
|---|------|-------------|----------|
| 2.1 | Anomaly heatmap generation | Write Colab cell to save anomaly score heatmaps as PNG for representative subset per dataset | 2 hr coding |
| 2.2 | Visual inspection | Manually verify that high-score regions correspond to actual anomalous objects, not artifacts | 1-2 hr |
| 2.3 | GT mask aspect ratio fix | Implement aspect-ratio-preserving resize with padding in load_ood_mask, re-run affected Large 1024 results | 2-3 hr coding + 2 hr runtime |
| 2.4 | Result consistency check | Cross-check all JSON results against sweep JSON, verify no duplicates or corrupt runs remain | 1 hr |
| **Block 2 total** | | | **8-10 hrs** |

### Block 3 — EoMT COCO Anomaly Segmentation (Step 8 completion)

| # | Task | Description | Estimate |
|---|------|-------------|----------|
| 3.1 | Run EoMT COCO on RA21 | Existing runner, swap checkpoint to eomt_coco.bin, num_classes=133 | 30 min runtime |
| 3.2 | Run EoMT COCO on RO21 | Same | 30 min runtime |
| 3.3 | Run EoMT COCO on LAF | Same, larger dataset | 1 hr runtime |
| 3.4 | Run EoMT COCO on fs_static | Same | 30 min runtime |
| 3.5 | Run EoMT COCO on RA | Same | 1 hr runtime |
| 3.6 | Temperature scaling sweep COCO | Offline sweep from cached logits on all datasets | 2-3 hr runtime |
| 3.7 | Collect and compare results | Update all_results.json, compare COCO vs Cityscapes checkpoint | 1 hr |
| **Block 3 total** | | | **6-8 hrs** |

### Block 4 — Fine-tuning EoMT COCO on Cityscapes (Step 5)

| # | Task | Description | Estimate |
|---|------|-------------|----------|
| 4.1 | Download Cityscapes train split | ~11 GB, extract training images and labels | 1-2 hr |
| 4.2 | Implement training loop | DataLoader, loss function (cross-entropy on 19 classes), optimizer, AMP scaler | 4-6 hr coding |
| 4.3 | Implement checkpoint saving | Save every N epochs, resume from checkpoint on Colab crash | 1-2 hr coding |
| 4.4 | Implement loss monitoring | CSV or TensorBoard logging per step and per epoch | 1 hr coding |
| 4.5 | Dry run — head only | Fine-tune prediction head only, freeze backbone, verify convergence on few batches | 2-3 hr coding + runtime |
| 4.6 | Full training run — head only | Train prediction head for 10-20 epochs, evaluate mIoU | 8-12 hr runtime |
| 4.7 | Gradual unfreezing experiment | Unfreeze last N backbone layers, compare mIoU vs head-only | 4-6 hr runtime |
| 4.8 | LoRA if needed | Implement LoRA adapters if full fine-tuning causes OOM on Colab A100 | 4-6 hr coding |
| 4.9 | Final mIoU evaluation | Evaluate best fine-tuned checkpoint, compare all three checkpoints | 1 hr |
| **Block 4 total** | | | **26-44 hrs** |

### Block 5 — Fine-tuned Model Anomaly Segmentation

| # | Task | Description | Estimate |
|---|------|-------------|----------|
| 5.1 | Run fine-tuned on all 5 datasets | Same runner, swap to fine-tuned checkpoint | 3-4 hr runtime |
| 5.2 | Temperature scaling sweep fine-tuned | Offline sweep from cached logits | 2-3 hr runtime |
| 5.3 | Three-checkpoint comparison | Compile full comparison table: Cityscapes vs COCO vs fine-tuned | 1 hr |
| **Block 5 total** | | | **6-8 hrs** |

### Block 6 — Report (5 pages)

| # | Task | Description | Estimate |
|---|------|-------------|----------|
| 6.1 | Introduction | Problem statement, motivation, paper overview | 2-3 hr |
| 6.2 | Related Work | ERFNet, EoMT, post-hoc methods, datasets | 3-4 hr |
| 6.3 | Methods | Pipeline description, post-hoc methods, temperature scaling, fine-tuning strategy | 3-4 hr |
| 6.4 | Results — anomaly segmentation | Tables, analysis, comparison across models and methods | 3-4 hr |
| 6.5 | Results — mIoU and fine-tuning | Three-checkpoint comparison, convergence analysis | 2-3 hr |
| 6.6 | Qualitative figures | Anomaly heatmaps, segmentation overlays, panoptic visualization | 1-2 hr |
| 6.7 | Conclusions | Summary of findings, limitations, future work | 1 hr |
| 6.8 | Revision and formatting | LaTeX formatting, table alignment, bibliography | 2-3 hr |
| **Block 6 total** | | | **17-24 hrs** |

---

## Grand Total

| Block | Task | Estimate |
|-------|------|----------|
| Block 1 | Cityscapes mIoU evaluation | 4-6 hrs |
| Block 2 | Result validation | 8-10 hrs |
| Block 3 | EoMT COCO anomaly segmentation | 6-8 hrs |
| Block 4 | Fine-tuning EoMT COCO | 26-44 hrs |
| Block 5 | Fine-tuned anomaly segmentation | 6-8 hrs |
| Block 6 | Report | 17-24 hrs |
| **Total** | | **67-100 hrs** |

At 6 to 8 hours per day full time, this corresponds to approximately 9 to 17 working days. Given the June 10 deadline, work should begin immediately with Block 1 and Block 3 in parallel, while Block 4 (fine-tuning) is started as early as possible due to its uncertain convergence timeline. Block 6 can be written incrementally alongside the other blocks as results become available.
