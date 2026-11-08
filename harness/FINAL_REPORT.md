# Final engineering checkpoint

**Completion status: PARTIAL, blocked on required source inputs.** The software foundation and synthetic integration paths are implemented and unit verified. The project is **not** scientifically complete or `PROJECT_VERIFIED`: no AGAR image/annotation package was present, and the requested original `PROJECT_SPEC.md` was missing. The checked-in requirements baseline is reconstructed from the task prompt.

## Verified work

| Area | Actual evidence | Status |
|---|---|---|
| Environment | Windows 11, isolated Python 3.12.13, PyTorch 2.14.0+cu130, RTX 5050 Laptop GPU, compute capability 12.0. CUDA matrix operation passed. `pip check` clean. | Verified |
| Data tooling | Generic COCO parser validates paths, image dimensions, boxes, IDs, categories, exact-file hashes, and empty records. Group/count-bin split and exact duplicate co-assignment tested. | Unit verified on synthetic fixtures |
| Classical CV | Configurable ROI, illumination, contrast, threshold, morphology, distance transform, watershed, area correction, boxes/counts, diagnostics, CLI, and validation-only Optuna runner. | Unit verified on synthetic fixtures; no AGAR inference or tuning |
| YOLO | One-epoch YOLO11n smoke on generated toy disks; checkpoint save/reload and tiled inference on RTX 5050. Split inference runner, score serialization, whole/tiled controls, and dataset conversion tested. | Synthetic smoke verified; no AGAR training |
| Evaluation | Count metrics, maximum-cardinality single-threshold matching, confidence-ranked 101-point AP, separate MAE/F1 threshold selection, paired image and group bootstrap, Wilcoxon, strata, plots, and saved-output ablation comparisons. | Unit verified on known synthetic records |
| Demo | Gradio 6.30.0 Blocks UI built; classical callback tested; missing YOLO checkpoint reported explicitly; loopback HTTP startup returned 200 and was stopped. | Local smoke verified; no AGAR YOLO model |
| Engineering gates | `pytest`: 58 passed in Python 3.12.13 GPU env and 58 passed in Python 3.14.5 system env. Ruff passed in both. `colony-counting verify`: `UNIT_VERIFIED`, `project_verified=false`. | Verified at unit level |

## Data and experiments actually used

- **AGAR data used:** None. The workspace had no AGAR images, annotations, manifest, or trained checkpoint. Raw AGAR schema, accessible sample size, lighting labels, plate groups, and duplicate rate are `NOT MEASURED`.
- **Synthetic software fixtures:** Generated disks, tiny saved boxes/predictions, and a one-image Optuna validation test were used for code-path checks only. Their scores are not research results.
- **GPU experiment:** One YOLO11n epoch on three generated 96x96 disk images, seed 42, image size 64, batch 2, AMP enabled, workers 0. Save/reload and tiled inference completed. A later inference refactor was exercised again on a generated toy image. No synthetic score or latency is presented as a benchmark result.
- **Thresholds, AGAR metrics, confidence intervals, p-values, species comparison, final test predictions, and ablation outcomes:** `NOT MEASURED`.

## Hardware and software

- Windows 11, NVIDIA GeForce RTX 5050 Laptop GPU, 8,151 MiB reported VRAM, driver 592.82, capability 12.0.
- Isolated `.venv-ml`: CPython 3.12.13, torch 2.14.0+cu130, torchvision 0.29.0+cu130, Ultralytics 8.4.183, OpenCV 4.14.0.94, Gradio 6.30.0.
- The system CPython 3.14.5 remains CPU-only for PyTorch. Resolved Windows packages are in `requirements-gpu-cu130.lock`.

## Outstanding work and limitations

- Obtain the original `PROJECT_SPEC.md` if it exists and an authorized local AGAR package. The official source describes CC BY-NC 2.0 academic research access and registration for the full dataset; no registration or download was performed.
- Inspect the source schema and licensing, verify image/category/lighting/camera fields, compute exact and perceptual duplicates, establish plate grouping, freeze a group-safe split, and generate real-data EDA.
- Run classical validation tuning and saved predictions; run YOLO training/checkpoint selection, whole-image versus tiled evaluation, and validation-only MAE/F1 threshold selection. Then freeze parameters and evaluate the final test partition once.
- Generate real mAP/counting metrics, paired confidence intervals and tests, required ablations, stratified tables, reviewed failure gallery, and a YOLO-backed demo check. ONNX comparison remains untested.
- Generic COCO normalization is not confirmed as the raw AGAR format. Exact-byte hashing does not catch resized/re-encoded near duplicates. Image-level bootstrap alone does not model plate dependence; use authoritative group IDs with the implemented cluster bootstrap when available.
- The exact original matching criteria, tolerance definitions, and requested count bins cannot be recovered from the missing specification. Current matching and count bins are explicit configurable/reconstructed choices.

## Reproduction commands

```powershell
.\.venv-ml\Scripts\python.exe -m pytest -q
.\.venv-ml\Scripts\ruff.exe check src tests scripts
.\.venv-ml\Scripts\python.exe -m colony_counting verify
.\.venv-ml\Scripts\python.exe scripts\synthetic_yolo_smoke.py
.\.venv-ml\Scripts\python.exe -m colony_counting app --config configs\classical.yaml
```

When authorized data is placed locally, use the data-intake, validation-only tuning, split prediction, threshold selection, evaluation, analysis, and ablation commands documented in `README.md` and `harness/RUNBOOK.md`. Do not use the test split for model or threshold selection.

## Resume bullets

- Built a configurable OpenCV colony-segmentation pipeline with diagnostic stages, validation-only Optuna tuning, and split-aware prediction export.
- Implemented and GPU-smoke-tested a tiled YOLO training/inference path with confidence-preserving predictions, threshold selection, and paired evaluation tooling; the smoke data were synthetic and no AGAR performance claim is available.
