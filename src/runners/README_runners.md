# `src/runners/`

Entry points for model inference and offline temperature sweep evaluation.
All runners produce structured artifacts and support reproducible execution via `--mode` and `--seed`.

---

## `run_erfnet_eval.py` — ERFNet inference runner

Evaluates ERFNet on anomaly segmentation datasets.

**Model:** ERFNet (`NUM_CLASSES=20` — 19 Cityscapes classes + `unlabelled`).
**Input resolution:** 1024×512 (Cityscapes standard).
**Methods:** `msp`, `maxentropy`, `maxlogit`.

**Anomaly scoring (`anomaly_from_logits`):**
- `maxlogit` — `-max(logits)` — temperature-invariant, operates on raw pre-softmax logits.
- `msp` — `1 - max(softmax(logits/T))`.
- `maxentropy` — `-Σ p·log p` where `p = softmax(logits/T)`.

**Usage:**
```bash
python3 -m src.runners.run_erfnet_eval \
  --input   "/path/to/dataset/images/*.*" \
  --weights "trained_models/erfnet_pretrained.pth" \
  --dataset-name RA21 \
  --method  msp \
  --temperature 1.0 \
  --mode    robust \
  --seed    0 \
  --deterministic \
  --artifacts-dir artifacts \
  --save-logits       # cache logits for offline sweep
```

**`--save-logits`** saves raw logits `[N, 20, 512, 1024]` as float32 `.npy` files enabling offline temperature sweep without rerunning inference.

---

## `run_eomt_eval.py` — EoMT inference runner

Evaluates EoMT on anomaly segmentation datasets.

**Model:** EoMT with DINOv2 ViT-Base backbone (`num_classes=19`).
**Input resolution:** 640×640 (default) — override with `--resize HxW`.
**Logit resolution:** 160×160 (4× downsampled from input).
**Methods:** `msp`, `maxentropy`, `maxlogit`, `rba`.

Anomaly maps are upsampled back to input resolution via bilinear interpolation before metric computation.

All post-processing imported from `src.utils.eomt_post` — see its README for formulas.

**Usage:**
```bash
python3 -m src.runners.run_eomt_eval \
  --input   "/path/to/dataset/images/*.*" \
  --ckpt    "path/to/eomt_cityscapes.bin" \
  --config  "eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" \
  --dataset-name RA21 \
  --method  rba \
  --temperature 1.0 \
  --mode    robust \
  --seed    0 \
  --deterministic \
  --artifacts-dir artifacts \
  --save-logits       # cache mask/class logits for offline sweep
```

**`--save-logits`** saves:
- `mask_logits` as float16 `[N, Q, h, w]`
- `class_logits` as float16 `[N, Q, C+1]`

Float16 halves disk usage with negligible precision loss for anomaly scoring.

---

## `sweep_temp_from_cache.py` — ERFNet temperature sweep

Offline temperature scaling sweep using cached ERFNet logits.
No model forward pass — loads `.npy` files and recomputes metrics at each temperature value.

**Requires:** `--save-logits` was used in `run_erfnet_eval.py`.

**Pro tip from project spec:** Save logits once, sweep temperatures instantly — avoid re-running the full model forward pass for each temperature value.

```bash
python3 -m src.runners.sweep_temp_from_cache \
  --dataset-name RA21 \
  --artifacts-dir artifacts \
  --use-latest \          # auto-resolves most recent run with valid logit cache
  --method msp \
  --mode   robust
```

**`--use-latest`** uses `resolve_latest_run_dir_filtered` from `artifacts.py` with `require_logits=True` — only selects runs that actually have the logit cache files on disk.

**Note on MaxLogit and temperature:** Temperature has no effect on MaxLogit by definition (operates on pre-softmax logits). All T values produce identical results — this is expected behavior.

---

## `sweep_temp_from_cache_eomt.py` — EoMT temperature sweep

Offline temperature sweep using cached EoMT mask/class logits.
Mirrors `sweep_temp_from_cache.py` but handles the dual-logit EoMT output.

Logits are loaded as float16 and cast to float32 for computation. Moved to device **once** before the temperature loop to avoid repeated GPU transfers.

```bash
python3 -m src.runners.sweep_temp_from_cache_eomt \
  --dataset-name RA21 \
  --artifacts-dir artifacts \
  --use-latest \
  --method rba \
  --mode   robust
```

**Default temperatures:** `0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0` — override with `--temperatures`.

---

## Execution modes (`--mode`)

| Mode | cuDNN | FPR95 impl | Weight loading | Use case |
|------|-------|------------|----------------|----------|
| `robust` | deterministic | discrete sweep | fuzzy (shape-matched) | reproducible results |
| `prof-exact` | benchmark=True | sklearn ROC | strict | reproduce instructor baseline |

---

## Artifact output structure

Each run produces:
```
artifacts/<dataset>/<model>/<timestamp>__<method>__T<T>__<mode>__<hash8>/
├── config.json
├── results/metrics.json
├── results/metrics.csv
├── logits/               (if --save-logits)
└── sweep/<method>__<mode>/
    ├── T<value>__metrics.json  (one per temperature)
    └── metrics_sweep.csv
```

The 8-character hash in the folder name is derived from the full run configuration — identical configs produce the same hash, making duplicate detection trivial.
