# `src/utils/`

Shared utility modules used by all runners and sweep scripts.
Each module has a single responsibility — no cross-dependencies between them.

---

## `ood_dataset.py` — GT mask loading and remapping

Central source of truth for ground-truth mask handling across all datasets.

**Key functions:**
- `gt_path_from_image(path_img)` — derives the GT mask path from an image path by replacing `images/` → `labels_masks/` and forcing `.png` extension for all supported datasets.
- `remap_ood_mask(path_gt, ood)` — remaps dataset-specific label encodings to the unified binary convention: `0 = InD`, `1 = OOD`, `255 = ignore/void`.
- `load_ood_mask(path_img, size_hw)` — loads, resizes (nearest-neighbour), and remaps in one call.
- `has_ood_pixels(ood)` — returns `True` if the mask contains at least one OOD pixel.

**Supported datasets and their remapping rules:**

| Dataset | Raw encoding | Remapping |
|---------|-------------|-----------|
| RoadAnomaly21 / RoadAnomaly | label 2 = OOD | `2 → 1` |
| RoadObstacle21 | already binary `{0,1}` | no-op |
| LostAndFound / FS_LostFound_full | legacy multi-class OR binary | guard + legacy map |
| Fishyscapes Static | already binary `{0,1}` | no-op |
| Streethazard | class 14 = anomaly | `14→void, <20→0, void→1` |

**Important fix — LostAndFound double-remap guard:**
Some LostAndFound exports are already in `{0, 1, 255}` binary format while others use the legacy multi-class encoding. `_is_already_binary_ood_mask()` detects this and skips the legacy remapping when the mask is already binary — preventing label inversion.

---

## `eomt_post.py` — EoMT post-processing and anomaly scoring

Single source of truth for all EoMT anomaly score computations.
All functions are imported by both `run_eomt_eval.py` and `sweep_temp_from_cache_eomt.py`.

**Key functions:**

- `pixel_probs_from_masks(mask_logits, class_logits, num_classes, temperature)`
  MaskFormer-style composition: `pixel[c,h,w] = Σ_q softmax(class_logits/T)[q,c] · sigmoid(mask_logits)[q,h,w]`, renormalized over classes. Used by MSP and MaxEntropy.

- `anomaly_from_pixel_probs(pixel_probs, method)`
  Computes MSP (`1 - max(P)`) or MaxEntropy (`-Σ P·log P`) from composed pixel probabilities.

- `anomaly_maxlogit_from_masks(mask_logits, class_logits, num_classes)`
  **Correct MaxLogit for mask architectures.** Aggregates raw class logits into pixel space weighted by mask probabilities — preserving the pre-softmax logit scale. Temperature has no effect by definition.
  Formula: `pixel_logits[c,h,w] = Σ_q sigmoid(mask_logits)[q,h,w] · class_logits[q,c]`, score = `1 - max_c(pixel_logits)`.

- `rba_from_masks(mask_logits, class_logits, num_classes, temperature, area_pow=0.5)`
  RbA (Rejected by All): reliability of each query = `conf · area^area_pow`, normality at pixel = `max_q(reliability · mask_prob)`, anomaly = `1 - normality`.

**Why MaxLogit is implemented separately from MSP/MaxEntropy:**
MSP and MaxEntropy require normalized pixel probabilities (softmax-composed). MaxLogit must operate on raw pre-softmax logits to avoid the overconfidence saturation problem. Using `log(pixel_probs)` as a proxy — as done in naive implementations — is mathematically incorrect and produces significantly worse results.

**`num_classes` convention:**
EoMT adds `+1` internally for the `no-object` token, so `class_head` shape = `(num_classes+1, 768)`. The checkpoint `eomt_cityscapes.bin` has `class_head=(20, 768)` → use `--num-classes 19`. All functions handle the `C+1` case automatically via the `if class_logits.shape[-1] == num_classes + 1` guard.

---

## `ood_metrics.py` — FPR@95TPR and AUPRC

Centralized metric computation. Single implementation used by all runners and sweeps.

**Functions:**
- `fpr_at_95_tpr(scores, labels, mode)` — dispatcher supporting two implementations:
  - `mode="robust"` (default): vectorized discrete threshold sweep via `cumsum` — deterministic, no interpolation, O(N log N).
  - `mode="roc"` / `"sklearn"`: sklearn `roc_curve` interpolation — matches some paper baselines but may differ slightly on pixel-level data.
- `fpr_at_95_tpr_sweep(scores, labels)` — vectorized implementation: sort scores descending, cumulative TP/FP, binary search for `target_tp = ceil(0.95 · P)`.
- `fpr_at_95_tpr_roc(scores, labels)` — sklearn-based fallback.

AUPRC is computed via `sklearn.metrics.average_precision_score` directly in runners.

---

## `artifacts.py` — experiment artifact management

Handles run directory creation, config saving, and run discovery for offline sweeps.

**Run directory naming convention:**
```
artifacts/<dataset>/<model>/<timestamp>__<method>__T<temperature>__<mode>__<hash8>/
├── config.json          # full run config for reproducibility
├── results/
│   ├── metrics.json
│   └── metrics.csv
├── logits/              # cached logits (optional, --save-logits)
│   ├── <dataset>__logits.npy          (ERFNet: float32 [N,C,H,W])
│   ├── <dataset>__mask_logits_f16.npy (EoMT: float16 [N,Q,h,w])
│   ├── <dataset>__class_logits_f16.npy
│   ├── <dataset>__gt.npy
│   └── <dataset>__names.json
└── sweep/               # temperature sweep results
    └── <method>__<mode>/
        ├── T<value>__metrics.json
        └── metrics_sweep.csv
```

**Key functions:**
- `create_run_dir(...)` — creates the full directory tree and saves `config.json`.
- `resolve_latest_run_dir_filtered(artifacts_root, dataset, model, method, mode, require_logits, logit_files)` — discovers the most recent run matching the given filters. Uses `RUN_RE` regex for robust structured name parsing. `require_logits=True` verifies that the expected cache files exist on disk before selecting a run — critical for sweep scripts that depend on cached logits.

---

## `determinism.py` — reproducibility control

Controls all sources of non-determinism in PyTorch.

**`apply_determinism(mode, seed, deterministic)`:**
- Sets Python, NumPy, and PyTorch seeds.
- Disables TF32 on Ampere GPUs (`allow_tf32=False`) to prevent floating-point differences across runs.
- `mode="robust"`: `cudnn.benchmark=False`, `cudnn.deterministic=True`, `torch.use_deterministic_algorithms(True, warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- `mode="prof-exact"`: `cudnn.benchmark=True` — matches instructor script behavior for baseline reproduction.
