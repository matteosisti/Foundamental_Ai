# Fine-tuning Notebooks

All outputs saved to `MyDrive/anom_project_private/` — separate from the shared Drive folder.

## Notebooks

### `step5_sweep_notturno.ipynb`
Full fine-tuning from COCO checkpoint.
- batch_size=32, 40 epochs, LR=1e-4, LLRD=0.9
- warmup_steps=[92, 184] (calibrated for bs=32)
- MaskFormer loss (BCE + Dice + CE + Hungarian matching)
- **Result: 64.03% mIoU** (SHA1: see `sweep_bs48/results/sweep_summary.json`)
- Outputs: `anom_project_private/sweep_bs48/`

### `step5_lora_on_sweep.ipynb`
LoRA r=16 stacked on top of the 64.03% sweep checkpoint.
- batch_size=32, 50 epochs, LR=5e-5
- LoRA targets: qkv + proj layers
- Starts from `sweep_bs48_ep40_final.bin`
- **Result: 64.24% mIoU** (SHA1: see `lora_on_sweep/results/lora_summary.json`)
- Outputs: `anom_project_private/lora_on_sweep/`

### `step5_lora_from_coco.ipynb`
LoRA r=16 directly from COCO checkpoint (no full FT first).
- batch_size=32, 30 epochs, LR=1e-4
- LoRA targets: qkv + proj layers
- Starts from `eomt_coco.bin`
- **Result: TBD** (in progress)
- Outputs: `anom_project_private/lora_from_coco/`

## Summary

| Method | Epochs | BS | Trainable params | mIoU ext |
|--------|--------|----|-----------------|----------|
| Full FT (sweep) | 40 | 32 | 93.5M | 64.03% |
| LoRA on sweep | 50 | 32 | ~2M | 64.24% |
| LoRA from COCO | 30 | 32 | ~2M | TBD |
| Stefano FT | 50 | 2 | 93.5M | 62.72% |
| Beybin FT | 50 | 2 | 93.5M | 61.71% |
| COCO zero-shot | — | — | — | 12.49% |
| CS baseline | — | — | — | 71.69% |

## Reproducibility
- Seed: 0 (fixed across all runs)
- All checkpoints on Drive with SHA1 documented
- WandB project: `anom-ft-private`
- Git clone from main branch (not ZIP)