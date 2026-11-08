# 🧫 Microbial Colony Counter

> **A reproducible benchmark comparing classical computer vision against deep learning (YOLO) for counting microbial colonies on agar plates — built on the [AGAR dataset](https://www.kaggle.com/datasets/ahsanshakoor/agar-bright-data?resource=download).**

---

## What is this?

Counting colonies on agar plates is a fundamental task in microbiology, yet it's still done manually in most labs — slow, tedious, and prone to human inconsistency. This project answers the question:

> *How much better (or worse) is a modern deep learning detector compared to a classical image-processing pipeline when counting microbial colonies automatically?*

We implement **both approaches end-to-end**, evaluate them against a shared ground truth from the AGAR dataset, run rigorous statistical comparisons (bootstrap + Wilcoxon), and ship a local **Gradio demo** so you can point either method at any agar plate image and get a live count.

---

## Why AGAR?

The **AGAR (Annotated Germs for Automated Recognition)** dataset, available on [Kaggle](https://www.kaggle.com/datasets/ahsanshakoor/agar-bright-data?resource=download), is one of the largest publicly available annotated datasets for microbial colony detection. It includes:

- **Countable** images — plates with labeled colony bounding boxes
- **Empty** images — plates with no colonies
- **Uncountable** images — overcrowded or otherwise unsuitable plates

This project uses a validated 70 / 15 / 15 **train / val / test split** with group-aware assignment (images from the same physical plate stay together), so there's no data leakage anywhere near your test set.

---

## Approaches compared

### 🔬 Classical Computer Vision Pipeline

A fully configurable, interpretable pipeline — no neural network, just pixels and math:

1. **Plate & ROI detection** — Hough circle transform to isolate the plate and exclude the rim
2. **Illumination correction** — Gaussian background subtraction to flatten uneven lighting
3. **CLAHE contrast enhancement** — adaptive histogram equalization for better colony visibility
4. **Thresholding** — Otsu or adaptive thresholding, configurable polarity (dark/bright colonies)
5. **Morphological cleanup** — opening and closing to remove noise and fill gaps
6. **Distance transform + watershed** — separates touching colonies
7. **Cluster correction** — area-based splitting for clumped colonies
8. **Diagnostics** — saves intermediate images at every stage so you can see exactly what happened

Every parameter lives in [`configs/classical.yaml`](configs/classical.yaml) and can be tuned automatically on the validation split using **Optuna** (never the test set).

### 🤖 Deep Learning — YOLOv11

A class-agnostic object detector fine-tuned on AGAR colony bounding boxes:

- **Tiled inference** — high-resolution plates are split into overlapping tiles; detections are merged back into global coordinates with NMS deduplication
- **Species-aware mode** — optional per-species class IDs if the dataset's taxonomy matters
- **Validation-selected thresholds** — separate confidence cutoffs are chosen on the validation split to minimize count MAE and maximize detection F1 independently
- **ONNX export** — optional export for deployment without a PyTorch dependency
- **GPU-accelerated** on NVIDIA cards via CUDA 13.0 (tested on RTX 5050 Laptop GPU)

---

## Project structure

```
.
├── src/colony_counting/
│   ├── data/
│   │   ├── agar.py          ← Native AGAR per-image JSON parser
│   │   ├── coco.py          ← COCO JSON parser + manifest writer
│   │   ├── splits.py        ← Deterministic group-aware 70/15/15 splits
│   │   └── eda.py           ← Dataset stats and diagnostic plots
│   ├── classical.py         ← End-to-end classical CV pipeline
│   ├── detection/
│   │   ├── yolo.py          ← YOLO training, inference, and checkpoint management
│   │   ├── yolo_data.py     ← COCO → YOLO label conversion
│   │   └── tiling.py        ← Tile geometry and global-coordinate merging
│   ├── manifest_inference.py← Frozen-split batch prediction runners
│   ├── evaluation.py        ← Counting + detection metrics, bootstrap, Wilcoxon, AP
│   ├── tuning.py            ← Validation-only Optuna parameter search
│   ├── analysis.py          ← Error plots, Bland-Altman, failure galleries
│   ├── app.py               ← Local Gradio demo
│   └── cli.py               ← Unified command-line interface
├── configs/
│   └── classical.yaml       ← All classical pipeline parameters
├── tests/                   ← 58 unit + integration tests
├── scripts/
│   └── synthetic_yolo_smoke.py ← One-epoch synthetic YOLO smoke test
├── harness/                 ← Experiment harness, state, plan, runbook
├── docs/                    ← Experiment records, limitations, progress
└── pyproject.toml
```

---

## Quickstart

### 1. Prerequisites

- Python **3.11 or 3.12** (3.12.13 recommended — that's the verified GPU environment)
- NVIDIA GPU with CUDA 13.0 support for YOLO training/inference (CPU works for classical and inference-only)
- The AGAR dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/ahsanshakoor/agar-bright-data?resource=download)

### 2. Set up the environment

**GPU environment (recommended — tested on Windows 11, RTX 5050, CUDA 13.0):**

```powershell
# Use the pinned CPython 3.12.13 for the verified GPU stack
& 'C:\Users\<you>\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe' -m venv .venv-ml

# Install pinned GPU dependencies
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-gpu-cu130.lock

# Install this package in editable mode
.\.venv-ml\Scripts\python.exe -m pip install -e . --no-deps
```

**Minimal CPU-only environment:**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[analysis,dev]"
```

### 3. Verify the install

```powershell
python -m pytest                          # 58 tests should pass
python -m colony_counting smoke           # core imports check
python -m colony_counting verify          # full gate report → results/verification.json
```

---

## Working with the AGAR dataset

### Parse the native AGAR layout

The AGAR dataset organizes images into `countable/`, `empty/`, and `uncountable/` subdirectories, each with a per-image JSON annotation file.

```powershell
# Build a validated manifest with frozen 70/15/15 splits
python -m colony_counting agar-data \
    --data-root path\to\agar_dataset \
    --output data\processed\manifest.jsonl \
    --seed 42
```

### Explore the data

```powershell
python -m colony_counting eda \
    --annotations path\to\annotations.json \
    --images path\to\images \
    --output results\eda
```

### Validate annotations

```powershell
python -m colony_counting validate-data \
    --annotations path\to\annotations.json \
    --images path\to\images
```

---

## Running the classical pipeline

### Count colonies in a single image

```powershell
python -m colony_counting classical \
    --image path\to\plate.jpg \
    --config configs\classical.yaml \
    --output results\classical
```

This writes a `prediction.json` with count, bounding boxes, and a `diagnostics/` folder with intermediate images for every stage of the pipeline.

### Generate predictions for a whole split

```powershell
python -m colony_counting predict-classical \
    --manifest data\processed\manifest.jsonl \
    --images-root path\to\images \
    --split val \
    --config configs\classical.yaml \
    --output results\predictions\classical-val.jsonl
```

### Tune parameters on the validation set (never test!)

```powershell
python -m colony_counting tune-classical \
    --manifest data\processed\manifest.jsonl \
    --images-root path\to\images \
    --config configs\classical.yaml \
    --trials 50 \
    --seed 42 \
    --output results\classical-tuning
```

Optuna searches over configurable hyperparameters and writes the best config, full trial CSV history, and provenance metadata. It is hard-locked to the `val` split.

---

## Running the YOLO pipeline

### Prepare YOLO-format labels from the AGAR manifest

```powershell
python -m colony_counting agar-prepare-yolo \
    --manifest data\processed\manifest.jsonl \
    --images-root path\to\images \
    --output data\yolo\colony
```

Add `--species-aware` for per-species class IDs instead of the class-agnostic `colony` label.

### Train

```powershell
python -m colony_counting train-yolo \
    --data data\yolo\colony\dataset.yaml \
    --weights weights\yolo11n.pt \
    --epochs 50 \
    --batch-size 4 \
    --seed 42 \
    --output results\yolo
```

### Infer on a single image

```powershell
python -m colony_counting infer-yolo \
    --image path\to\plate.jpg \
    --weights results\yolo\colony-yolo\weights\best.pt \
    --confidence 0.25 \
    --output results\yolo\inference
```

Tiled inference is on by default (`--whole-image` to disable). The overlay and a JSON prediction file are saved to `--output`.

### Generate predictions for a whole split

```powershell
python -m colony_counting predict-yolo \
    --manifest data\processed\manifest.jsonl \
    --images-root path\to\images \
    --split val \
    --checkpoint results\yolo\colony-yolo\weights\best.pt \
    --confidence-floor 0.001 \
    --output results\predictions\yolo-val.jsonl
```

### Select confidence thresholds on validation data

```powershell
python -m colony_counting tune-thresholds \
    --predictions results\predictions\yolo-val.jsonl \
    --manifest data\processed\manifest.jsonl \
    --output results\yolo-thresholds.json
```

Writes two independent thresholds — one minimizing count MAE, one maximizing detection F1 — from validation predictions only.

---

## Evaluation and analysis

### Evaluate saved predictions

```powershell
python -m colony_counting evaluate \
    --predictions results\predictions\classical-test.jsonl \
    --match-mode iou \
    --match-threshold 0.5 \
    --output results\evaluation-classical.json
```

Optionally compare two methods directly:

```powershell
python -m colony_counting evaluate \
    --predictions results\predictions\classical-test.jsonl \
    --compare results\predictions\yolo-test.jsonl \
    --match-mode iou \
    --match-threshold 0.5 \
    --bootstrap-resamples 2000 \
    --seed 42 \
    --output results\comparison.json
```

Metrics reported:
- **Counting:** MAE, RMSE, within-tolerance accuracy (absolute and relative), over/under-count rates
- **Detection:** Precision, Recall, F1 (one-to-one matching), mean AP
- **Statistical tests:** Paired bootstrap CI on MAE difference (2,000 resamples), paired Wilcoxon signed-rank test

### Error analysis and failure gallery

```powershell
python -m colony_counting analyze \
    --predictions results\predictions\classical-test.jsonl \
    --manifest data\processed\manifest.jsonl \
    --split test \
    --images-root path\to\images \
    --output results\analysis
```

Writes count agreement plots, Bland-Altman plots, stratified breakdowns, and a worst-case image gallery.

### Ablation comparisons

```powershell
python -m colony_counting ablations \
    --baseline results\predictions\baseline.jsonl \
    --variant tuned=results\predictions\tuned.jsonl \
    --manifest data\processed\manifest.jsonl \
    --split test \
    --output results\ablations.json
```

---

## Local demo (Gradio)

```powershell
# Classical-only demo (no checkpoint needed)
.\.venv-ml\Scripts\python.exe -m colony_counting app --config configs\classical.yaml

# With YOLO (requires a trained checkpoint)
.\.venv-ml\Scripts\python.exe -m colony_counting app \
    --config configs\classical.yaml \
    --checkpoint results\yolo\colony-yolo\weights\best.pt
```

Opens at **http://127.0.0.1:7860** — upload any agar plate image, choose classical or YOLO, and get a colony count with an annotated overlay and latency readout. Missing checkpoints raise an explicit error rather than silently returning wrong results.

---

## Classical pipeline configuration

All classical pipeline knobs live in [`configs/classical.yaml`](configs/classical.yaml):

| Parameter | Default | Description |
|---|---|---|
| `channel` | `gray` | Input channel (`gray`, `red`, `green`, `blue`) |
| `illumination_correction` | `true` | Gaussian background subtraction |
| `background_sigma` | `31.0` | Sigma for background estimation kernel |
| `clahe_clip_limit` | `2.0` | CLAHE contrast clip limit |
| `threshold_method` | `otsu` | `otsu` or `adaptive` |
| `polarity` | `dark` | Colony appearance (`dark` or `bright`) |
| `plate_detection` | `true` | Auto-detect circular plate boundary |
| `plate_roi_scale` | `0.92` | Fraction of plate radius to use as ROI |
| `peak_min_distance` | `7` | Minimum distance between colony centroids |
| `min_colony_area` | `8` | Minimum blob area in pixels |
| `max_colony_area` | `10000` | Maximum blob area in pixels |
| `cluster_area_correction` | `false` | Split large blobs by estimated colony size |
| `nominal_colony_area` | `300.0` | Expected single-colony area for cluster splitting |

---

## Development

```powershell
# Run tests
python -m pytest

# Lint
python -m ruff check src tests scripts

# Full project gate report
python -m colony_counting verify

# Synthetic YOLO smoke test (no AGAR data needed)
.\.venv-ml\Scripts\python.exe scripts\synthetic_yolo_smoke.py
```

The test suite covers: COCO/AGAR parsing, group-aware splits, duplicate detection, classical pipeline components, YOLO label conversion, tile geometry, evaluation metrics, one-to-one matching, AP computation, bootstrap resampling, Wilcoxon edge cases, Optuna tuning isolation, Gradio app callbacks, and CLI round-trips — all using synthetic fixtures with no AGAR images required.

---

## Current status

| Milestone | Status | Notes |
|---|---|---|
| Environment & tooling | ✅ Verified | Python 3.12.13, CUDA 13.0, RTX 5050, 58 tests passing |
| Data parsing & splits | ✅ Code verified | AGAR native parser + COCO adapter; awaiting real data run |
| Classical pipeline | ✅ Code verified | Full pipeline + Optuna tuning; awaiting real images |
| YOLO training & inference | ✅ Smoke verified | One-epoch synthetic smoke passed on GPU; awaiting AGAR training |
| Evaluation & statistics | ✅ Code verified | All metrics, bootstrap, Wilcoxon, AP; awaiting real predictions |
| Error analysis | ✅ Code verified | Plots + galleries; awaiting real predictions |
| Gradio demo | 🟡 In progress | Classical works; YOLO demo awaits trained checkpoint |
| Scientific results | ⏳ Blocked | Requires AGAR dataset + completed training runs |

> [!NOTE]
> All "Verified" labels above reflect **software verification on synthetic fixtures**. No scientific results exist yet — those require completing the training and inference runs on the full AGAR dataset. See [`docs/experiments.md`](docs/experiments.md) for the distinction between software tests and research evidence.

---

## Experiment integrity rules

This project is designed to be honest about what has and hasn't been measured:

- **Test data is never touched** until final evaluation — splits are frozen before any tuning
- **No fabricated metrics** — the experiment log explicitly notes when results are from synthetic fixtures vs. real AGAR data
- **Synthetic fixtures are labelled** — they test code behavior only and are never reported as research evidence
- **All predictions are saved** — evaluation is decoupled from inference, so results are always traceable to exact prediction files
- **Bootstrap uses a fixed seed** — 2,000 resamples, seed 42, for reproducibility

---

## Citation / Dataset

If you use this project, please cite the AGAR dataset:

```
Zieliński, B. et al. (2020). Deep learning approach to bacterial colony classification.
PLOS ONE. https://doi.org/10.1371/journal.pone.0234312
```

Dataset: https://www.kaggle.com/datasets/ahsanshakoor/agar-bright-data?resource=download

---

## License

See repository license file. Dataset use is governed by the AGAR dataset terms on Kaggle.
