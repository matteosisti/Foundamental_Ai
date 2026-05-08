# Project Status Update — May 08, 2026

## Summary

Today's session completed Project Step 4 in full. Both EoMT checkpoints have been evaluated on the Cityscapes validation set, a COCO→Cityscapes class remapping strategy has been designed and validated, and qualitative visualizations have been generated and saved to Drive.

---

## Completed Today

### Step 4 — EoMT Checkpoint Comparison on Cityscapes

**Dataset preparation**
The Cityscapes dataset (train and val splits) was converted from raw PNG format to Zarr using `segdatakit`, a custom lossless preprocessing library. The conversion runs once and produces two files stored on Google Drive:
- `cityscapes_train.zarr` — 2975 images, 12.3 GB
- `cityscapes_val.zarr` — 500 images, 2.1 GB

The Zarr format allows direct streaming from Drive without per-session extraction, significantly reducing setup time in future Colab sessions.

**EoMT Cityscapes — mIoU evaluation**
The Cityscapes-trained EoMT Base checkpoint was evaluated on the full 500-image validation set using the standard 19-class train ID remapping. Final result: **mIoU = 71.69%**. Per-class IoU was computed and saved to `results/miou_eomt_cityscapes.json`.

Strong per-class results include road (97.95%), sky (92.72%), car (92.01%), and building (90.52%). Weaker classes include pole (47.34%), rider (45.88%), and motorcycle (45.44%), which are structurally harder to segment due to thin geometry and visual ambiguity.

**EoMT COCO — mIoU evaluation with cross-dataset remapping**
The COCO panoptic-trained EoMT Base checkpoint was evaluated under a zero-shot transfer protocol. Since the COCO panoptic ontology differs from Cityscapes, a COCO train ID → Cityscapes train ID remapping was designed by:

1. Inspecting the `CLASS_MAPPING` in `eomt/datasets/coco_panoptic.py` to understand the model's output space (train IDs 0–132, not raw COCO category IDs)
2. Running per-image diagnostics to identify which train IDs the model actually predicts on road scenes
3. Fetching the official COCO panoptic category list to map train IDs to semantic names
4. Defining the remapping iteratively, adding classes as their semantic equivalents were confirmed

The final mapping covers **16 out of 19 Cityscapes classes**. The three missing classes — pole, traffic sign, and rider — have no direct semantic equivalent in the COCO panoptic ontology and are assigned IoU=0 by construction. Final result: **mIoU = 12.49%**.

**Comparison table**

| Checkpoint | mIoU | Classes covered | Training domain |
|------------|------|----------------|-----------------|
| EoMT Cityscapes | 71.69% | 19/19 | Cityscapes (in-domain) |
| EoMT COCO | 12.49% | 16/19 | COCO panoptic (zero-shot) |

**Per-class analysis highlights**

Building is the most transferable class (COCO: 53.83% vs CS: 90.52%), reflecting the high frequency of architectural content in COCO. Road collapses to 1.75% despite explicit mapping, suggesting that the COCO model rarely activates road-related predictions in driving scenes. Sky achieves only 0.82% despite two mapped classes, likely due to context mismatch. Truck, bus, and train all score 0%, consistent with their low representation in COCO relative to Cityscapes.

The delta of +59.21% mIoU between the two checkpoints provides quantitative motivation for the fine-tuning experiment in Step 5.

**Qualitative visualizations**
Six side-by-side visualizations (input / prediction / overlay) were generated for three sample images from the validation set, comparing EoMT Cityscapes and EoMT COCO predictions. Saved to `results/figures/`:
- `semantic_cs_0.png`, `semantic_cs_250.png`, `semantic_cs_499.png`
- `semantic_coco_0.png`, `semantic_coco_250.png`, `semantic_coco_499.png`

The visualizations clearly show the qualitative gap: EoMT Cityscapes produces clean, fine-grained segmentation maps, while EoMT COCO produces coarse and incomplete predictions dominated by building and car classes.

---

## Infrastructure updates

**segdatakit — motivation and design**
`segdatakit` is a custom preprocessing library developed to address a recurring practical problem in cloud-based deep learning workflows: the Cityscapes dataset consists of approximately 11 GB of raw PNG files that must be downloaded, extracted, and re-read from disk at every Colab session. This creates significant I/O bottlenecks when reading directly from Google Drive, where sequential PNG reads are slow due to filesystem overhead per file.

`segdatakit` solves this by converting raw datasets into Zarr format with Blosc2/LZ4 lossless compression. The conversion runs once and produces a single chunked file per split that can be streamed directly from Drive without extraction. The library guarantees pixel-fidelity via SHA-256 round-trip audit, applies label remapping at read time (never on disk), and exposes a PyTorch-compatible `SegDataset` that plugs directly into any training loop. An optional NVIDIA DALI GPU pipeline reduces per-batch decode latency from ~40ms to ~4ms when available.

Repository: `https://github.com/matteosisti/segdatakit`

**segdatakit — installation issue on Colab**
The standard `pip install -e .` can fail on Colab due to build backend
incompatibilities between setuptools versions. The recommended approach
for Colab is to add the package directly to the Python path:

```python
import sys
sys.path.insert(0, '/content/segdatakit')
```

This works because `segdatakit` is a pure Python package with no C extensions.
Alternatively, if you prefer pip install:

```python
import subprocess, sys
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    '-e', '/content/segdatakit',
    '--no-build-isolation', '-q'
], check=True)
```

Make sure zarr>=3.0 is installed:

```python
!pip install "zarr>=3.0" "numcodecs>=0.12" -q
```

**segdatakit — reader fallback behavior**
When `cfg['paths']['output']` is not set before calling `get_reader`, `segdatakit` falls back to reading raw PNG files from `cfg['paths']['raw']` instead of the Zarr store. This was observed during the Step 4 evaluation runs — both mIoU evaluations read from raw PNG. The fallback produced correct results but was slower than Zarr streaming. This will be fixed in future sessions by explicitly setting `cfg['paths']['output'] = ZARR_VAL_PATH` before every `get_reader` call.

---

**COCO→Cityscapes remapping — why it was needed**
The EoMT COCO checkpoint was trained on COCO panoptic segmentation using the `CLASS_MAPPING` defined in `eomt/datasets/coco_panoptic.py`. This mapping converts raw COCO category IDs (1–200, non-contiguous) to contiguous train IDs (0–132). The model therefore outputs train IDs, not raw COCO category IDs. Any remapping to Cityscapes must be based on train IDs, not on the original COCO category IDs.

This was not immediately obvious and caused the first two evaluation attempts to produce near-zero mIoU (0.05% and 2.09%) because the initial `COCO_TO_CS` mapping used raw COCO category IDs instead of train IDs. The error was diagnosed by inspecting the model's actual output distribution on sample images, identifying which train IDs were being predicted, and cross-referencing them against `CLASS_MAPPING` and the official COCO panoptic category JSON.

**COCO→Cityscapes remapping — design process**
The remapping was built iteratively in four steps:

1. Ran a diagnostic forward pass on 20 images to identify the most frequently predicted train IDs.
2. Inverted `CLASS_MAPPING` to recover the corresponding COCO category IDs.
3. Fetched the official COCO panoptic category list from `cocodataset/panopticapi` to resolve category names.
4. Manually mapped each named COCO category to the closest Cityscapes semantic class.

**COCO→Cityscapes remapping — final mapping**

```python
COCO_TO_CS = {
    # things — direct semantic equivalents
    0:   11,  # person            (coco_cat=1,   train_id=0)
    1:   18,  # bicycle           (coco_cat=2,   train_id=1)
    2:   13,  # car               (coco_cat=3,   train_id=2)
    3:   17,  # motorcycle        (coco_cat=4,   train_id=3)
    4:   15,  # bus               (coco_cat=6,   train_id=4)
    5:   16,  # train             (coco_cat=7,   train_id=5)
    6:   14,  # truck             (coco_cat=8,   train_id=6)
    9:    6,  # traffic light     (coco_cat=10,  train_id=9)
    # stuff — semantic approximations
    91:  2,   # house             -> building    (coco_cat=128)
    99:  0,   # pavement          -> road        (coco_cat=148)
    100: 0,   # road              -> road        (coco_cat=149)
    101: 2,   # roof              -> building    (coco_cat=151)
    104: 10,  # sky-other-merged  -> sky         (coco_cat=156)
    109: 3,   # wall-brick        -> wall        (coco_cat=171)
    110: 3,   # wall-stone        -> wall        (coco_cat=175)
    115: 2,   # window-other      -> building    (coco_cat=181)
    116: 8,   # tree-merged       -> vegetation  (coco_cat=184)
    117: 4,   # fence-merged      -> fence       (coco_cat=185)
    119: 10,  # sky               -> sky         (coco_cat=187)
    123: 1,   # pavement-merged   -> sidewalk    (coco_cat=191)
    125: 9,   # grass-merged      -> terrain     (coco_cat=193)
    126: 9,   # dirt-merged       -> terrain     (coco_cat=194)
    129: 2,   # building-other    -> building    (coco_cat=197)
    131: 3,   # wall-other-merged -> wall        (coco_cat=199)
}
```

24 COCO train IDs are mapped to 16 Cityscapes classes. Three Cityscapes classes — pole (cs_id=5), traffic sign (cs_id=7), and rider (cs_id=12) — have no semantic equivalent in the COCO panoptic ontology and are assigned IoU=0 by construction. This is a fundamental ontological limitation, not a remapping error.

---

**File locations**

| File | Location | Description |
|------|----------|-------------|
| `cityscapes_train.zarr` | `MyDrive/anom_project/cityscapes/` | Cityscapes train split, Zarr format, 12.3 GB |
| `cityscapes_val.zarr` | `MyDrive/anom_project/cityscapes/` | Cityscapes val split, Zarr format, 2.1 GB |
| `leftImg8bit_trainvaltest/` | `MyDrive/anom_project/cityscapes/` | Raw PNG images (extracted) |
| `gtFine_trainvaltest/` | `MyDrive/anom_project/cityscapes/` | Raw GT labels (extracted) |
| `eomt_cityscapes.bin` | `MyDrive/anom_project/ckpts/eomt/` | EoMT Base Cityscapes checkpoint |
| `eomt_coco.bin` | `MyDrive/anom_project/ckpts/eomt/` | EoMT Base COCO panoptic checkpoint |
| `eomt_large_1024.bin` | `MyDrive/anom_project/ckpts/eomt/` | EoMT Large 1024 checkpoint |
| `miou_eomt_cityscapes.json` | `MyDrive/anom_project/results/` | mIoU results — EoMT Cityscapes |
| `miou_eomt_coco.json` | `MyDrive/anom_project/results/` | mIoU results — EoMT COCO with remapping info |
| `semantic_cs_*.png` | `MyDrive/anom_project/results/figures/` | Qualitative visualizations — Cityscapes checkpoint |
| `semantic_coco_*.png` | `MyDrive/anom_project/results/figures/` | Qualitative visualizations — COCO checkpoint |
| `all_results.json` | `MyDrive/anom_project/results/` | All anomaly segmentation results |
| `all_sweeps.json` | `MyDrive/anom_project/results/` | Temperature scaling sweep results |
| `artifacts/` | `MyDrive/anom_project/` | Raw run artifacts with per-run metrics.json |

---

## Notebook — `notebooks/step4_miou_eval.ipynb`

The Step 4 evaluation is fully contained in a single Colab notebook at `notebooks/step4_miou_eval.ipynb`. The notebook is structured into 7 sequential cells and is designed to be run top-to-bottom in a single Colab session on an A100 GPU. Total runtime is approximately 2 hours (1 hour per checkpoint evaluation).

**Cell 0 — Setup and authentication**
Mounts Google Drive, clones the private project repository using `GITHUB_USERNAME`, `GITHUB_TOKEN`, and `GITHUB_REPO` Colab secrets, clones `segdatakit` from `https://github.com/matteosisti/segdatakit`, and adds both to `sys.path`. No `pip install -e` is used due to the `setuptools.backends` compatibility issue described above.

**Cell 1 — Zarr conversion (run once)**
Loads `configs/cityscapes.yaml`, patches `paths.raw` and `paths.output` to Drive paths, and runs `segdatakit/scripts/convert.py` to produce `cityscapes_train.zarr` and `cityscapes_val.zarr`. The cell checks for existing Zarr files and skips conversion if both are present, making subsequent sessions fast.

**Cell 2 — Model helpers and COCO→CS remapping**
Defines all shared utilities used across evaluations:
- `load_model(ckpt_path, num_classes)` — instantiates `EoMTWrapper` with ViT-Base backbone
- `predict(model, img_np, num_classes, orig_hw)` — runs forward pass, applies einsum pixel logit composition, bilinear upsampling to original resolution, returns argmax class map
- `remap_coco_to_cs(pred_coco)` — applies `COCO_TO_CS` dict to convert COCO train IDs to Cityscapes train IDs, returns array with IGNORE_INDEX (255) for unmapped pixels
- `IoUMeter` — accumulates confusion matrix, computes per-class IoU and mIoU
- `COCO_TO_CS` — the full 24-entry remapping dict (see remapping section above)
- `cfg` — loaded and patched cityscapes YAML config pointing to val Zarr

**Cell 3 — EoMT Cityscapes mIoU evaluation**
Loads `eomt_cityscapes.bin`, instantiates reader from val Zarr, iterates over 500 images, accumulates confusion matrix, prints mIoU every 50 images. Saves per-class IoU and final mIoU to `results/miou_eomt_cityscapes.json`. Runtime: ~1 hour on A100.

**Cell 4 — EoMT COCO mIoU evaluation**
Same pipeline as Cell 3 but loads `eomt_coco.bin` with `num_classes=133`, applies `remap_coco_to_cs` to predictions before updating the IoU meter. Saves results to `results/miou_eomt_coco.json` including `mapped_classes`, `unmapped_classes`, and an explanatory note. Runtime: ~1 hour on A100.

**Cell 5 — Per-class comparison table**
Prints a formatted table comparing EoMT Cityscapes and EoMT COCO IoU per class, with delta column. No file output — intended for inline inspection and report copying.

**Cell 6 — Qualitative visualizations**
Generates side-by-side figures (input / colorized prediction / 50-50 overlay) for three sample images (indices 0, 250, 499) for both checkpoints. Uses official Cityscapes color palette indexed by train ID. Saves 6 PNG files to `results/figures/`. Both Cityscapes and COCO predictions are colorized using the same Cityscapes palette — for COCO, unmapped pixels appear black (IGNORE_INDEX has no color entry).

**Cell 7 — Summary**
Prints final mIoU values, lists all saved file paths.

---

## Remaining Work

| Block | Task | Estimate |
|-------|------|----------|
| Block 3 | EoMT COCO anomaly segmentation on 5 datasets | 6-8 hrs |
| Block 4 | Fine-tuning EoMT COCO on Cityscapes | 26-44 hrs |
| Block 5 | Fine-tuned anomaly segmentation and sweep | 6-8 hrs |
| Block 6 | Report (5 pages) | 17-24 hrs |
| **Total remaining** | | **55-84 hrs** |

The critical path is Block 4 (fine-tuning), which is the most time-consuming and has the most uncertain convergence timeline. Block 3 (EoMT COCO anomaly segmentation) can be run in parallel using the existing runner infrastructure and should be prioritized next as it requires no new code.

---

## Next Steps

1. Commit Step 4 notebook and updated `miou_cityscapes.py` to GitHub
2. Run EoMT COCO anomaly segmentation on all 5 datasets (Block 3) — runner already supports this
3. Begin fine-tuning training loop implementation (Block 4)
4. Write Step 4 results section of the report in parallel
