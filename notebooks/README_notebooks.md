# `notebooks/`

Colab notebooks for running the full evaluation pipeline.

---

## `anomaly_eval_pipeline.ipynb`

End-to-end evaluation notebook for ERFNet and EoMT on all 5 anomaly segmentation benchmarks.
Designed to run on **Google Colab with T4 GPU** (minimum), **A100 recommended** for large datasets.

---

## Setup

### 1. Colab Secrets
Before running, add the following secrets in the Colab ** Secrets panel** (left sidebar):

| Secret name | Value |
|-------------|-------|
| `GITHUB_USERNAME` | your GitHub username |
| `GITHUB_TOKEN` | personal access token with repo scope |
| `GITHUB_REPO` | repository name (e.g. `Foundamental_Ai_private`) |
| `USER_EMAIL` | your git email |

The notebook reads them via:
```python
from google.colab import userdata
GITHUB_USER  = userdata.get("GITHUB_USERNAME")
GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")
```
Credentials never appear in the notebook output or git history.

### 2. Runtime
Set runtime to **T4 GPU** before running:
`Runtime → Change runtime type → T4 GPU`

For large datasets (LAF, 99 images) an **A100 is recommended** — the temperature sweep loads all logits into VRAM at once and may OOM on T4.

### 3. Drive structure
The notebook expects the following layout on Google Drive:

```
MyDrive/anom_project/
├── ckpts/
│   └── eomt/
│       └── eomt_cityscapes.bin       ← EoMT pretrained checkpoint
├── Validation_Dataset/
│   ├── RoadAnomaly21/images/
│   ├── RoadObstacle21/images/
│   ├── LostAndFound/images/
│   ├── fs_static/images/
│   └── RoadAnomaly/images/
└── artifacts/                        ← created automatically by runners
```

---

## Running the notebook

The notebook is structured in **independent cells per dataset** — you can run a single cell without re-running the full pipeline.

| Cell | Content |
|------|---------|
| 1 | GPU check |
| 2 | Secrets + Drive mount + Git clone/pull |
| 3 | Global config + `erfnet()` and `eomt()` helpers |
| 3.1 – 3.5 | ERFNet on each dataset |
| 4.1 – 4.5 | EoMT on each dataset |

Each dataset cell runs inference + temperature sweep in sequence.

---

## How the sweep auto-discovery works

Each runner saves artifacts under a timestamped directory:
```
artifacts/<dataset>/<model>/
  2026-03-31_12-30-13__msp__T1.0__robust__cdb52244/
  2026-03-31_12-31-18__maxlogit__T1.0__robust__cb1c6f88/
  ...
```

The sweep scripts use `--use-latest` which calls `resolve_latest_run_dir_filtered()` in `artifacts.py`. This function:

1. Lists all subdirectories under `artifacts/<dataset>/<model>/`
2. Parses each folder name with the regex:
   ```
   ^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})__
   (?P<method>[a-zA-Z0-9\-]+)__
   T(?P<T>[-+]?\d*\.?\d+)__
   (?P<mode>robust|prof-exact)__
   (?P<hash>[0-9a-fA-F]+)$
   ```
3. Filters by `method` and `mode`
4. With `require_logits=True`, verifies that the expected `.npy` cache files exist on disk:
   - ERFNet: `<dataset>__logits.npy` + `<dataset>__gt.npy`
   - EoMT: `<dataset>__mask_logits_f16.npy` + `<dataset>__class_logits_f16.npy` + `<dataset>__gt.npy`
5. Returns the **most recent** matching run (lexicographic sort on ISO timestamp)

This means you can safely rerun inference — the sweep always picks up the latest valid run automatically.

---

## Artifact output on Drive

All results are saved to `MyDrive/anom_project/artifacts/` and persist across Colab sessions.

```
artifacts/
└── <DATASET>/
    └── <MODEL>/
        └── <timestamp>__<method>__T<T>__<mode>__<hash8>/
            ├── config.json                  ← full run config
            ├── results/
            │   ├── metrics.json             ← AuPRC, FPR95, metadata
            │   └── metrics.csv
            ├── logits/                      ← cached logits (--save-logits)
            │   ├── <DATASET>__logits.npy              (ERFNet float32 [N,C,H,W])
            │   ├── <DATASET>__mask_logits_f16.npy     (EoMT float16 [N,Q,h,w])
            │   ├── <DATASET>__class_logits_f16.npy    (EoMT float16 [N,Q,C+1])
            │   ├── <DATASET>__gt.npy                  (uint8 [N,H,W])
            │   └── <DATASET>__names.json
            └── sweep/
                └── <method>__<mode>/
                    ├── T0.5__metrics.json
                    ├── T0.75__metrics.json
                    ├── ...
                    └── metrics_sweep.csv
```

> **Note:** Logit files are large (ERFNet LAF ≈ 4GB, EoMT ≈ hundreds of MB per dataset).
> Once you have all final results in `results/metrics.json`, the `logits/` folders can be deleted to free Drive space — all AuPRC and FPR95 numbers are already saved in the json files.

---

## Note on temperature scaling and MaxLogit

Temperature scaling has **no effect on MaxLogit** by design — MaxLogit operates on raw pre-softmax logits, which are scale-invariant with respect to temperature. All temperature values in the sweep will produce identical results for `maxlogit`. This is expected behavior, not a bug.