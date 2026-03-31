# `src/models/`

Model wrappers providing a unified inference interface for all architectures.

---

## `eomt_wrapper.py` — EoMT model wrapper

Robust wrapper for **EoMT** (Everything on Mask Transformer, CVPR 2025),
a ViT-based mask architecture using DINOv2 as backbone.

### Architecture

EoMT decomposes segmentation into two parallel outputs per decoder layer:
- `mask_logits` — `[B, Q, H, W]`: spatial occupancy of each of the `Q=100` learned queries
- `class_logits` — `[B, Q, C+1]`: class assignment probability for each query (C classes + no-object)

Only the **final decoder layer** outputs are used for inference (`mask_list[-1]`, `class_list[-1]`).

### Checkpoint loading — `num_classes` convention

EoMT adds `+1` to `num_classes` internally when building `class_head`:
- `num_classes=19` → `class_head=(20, 768)` ✅ matches `eomt_cityscapes.bin`
- `num_classes=20` → `class_head=(21, 768)` ❌ shape mismatch, weights not loaded

The single `missing=1` at load time is `encoder.backbone.pos_embed` — shape `(1, 4096, 768)` in the checkpoint (1024×1024 training resolution) vs `(1, 1600, 768)` at inference (640×640). EoMT handles this automatically via **positional embedding interpolation** at runtime.

### Fuzzy weight loading

`_load_weights_robust()` handles checkpoint format variability:
- Unwraps Lightning-style `{"state_dict": {...}}` checkpoints
- Strips common key prefixes: `network.`, `model.`, `module.`
- Loads only parameters with **compatible shapes** — skips mismatches silently
- Falls back to stripping `eomt.` prefix if no matches found on first pass

This makes the wrapper compatible with checkpoints saved in different formats without modifying original files.

### Import aliasing

EoMT's internal imports use bare module names (`from models.xxx import ...`) instead of fully qualified paths. `_alias_eomt_subpackages()` creates runtime aliases in `sys.modules` at instantiation time:
```
models   → eomt.models
datasets → eomt.datasets
utils    → eomt.utils
```
This avoids patching the original EoMT source files.

### Public API

```python
model = EoMTWrapper(
    img_size=(640, 640),
    num_classes=19,          # Cityscapes semantic classes
    num_q=100,               # number of learned mask queries
    num_blocks=3,            # decoder blocks
    backbone_name="vit_base_patch14_reg4_dinov2",
    masked_attn_enabled=True,
)
model.load(ckpt_path, device, mode="robust")  # or "prof-exact" for strict loading

mask_logits, class_logits = model.forward_masks_and_classes(x)
# mask_logits:  [B, Q, H, W]  — spatial mask occupancy
# class_logits: [B, Q, C+1]   — class scores per query
```

### Supported configurations

| Config file | Backbone | Resolution | Queries |
|-------------|----------|------------|---------|
| `eomt_base_640.yaml` | ViT-Base/14 DINOv2 | 640×640 | 100 |
| `eomt_large_640.yaml` | ViT-Large/14 DINOv2 | 640×640 | 100 |
| `eomt_base_1024.yaml` | ViT-Base/14 DINOv2 | 1024×1024 | 100 |

For 1024×1024, `pos_embed` loads correctly from the checkpoint without interpolation.
