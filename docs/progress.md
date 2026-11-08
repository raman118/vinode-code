# Progress

## 2026-10-21

- Inspected a clean workspace. `PROJECT_SPEC.md` was absent and no dataset/checkpoint files were found. Added a clearly labeled requirements baseline from the supplied master prompt.
- Audited the environment: Windows 11, Python 3.14.5, RTX 5050 (8,151 MiB reported VRAM), driver 592.82, 100.8 GB free disk, 23.7 GiB RAM. Installed torch is CPU-only (`2.12.0+cpu`); CUDA tensor execution is unavailable with the installed wheel.
- Confirmed core data/analysis packages are importable; Ultralytics and Gradio are absent.
- Implemented generic COCO input parsing, file/dimension/box validation, empty-image preservation, and atomic JSONL manifest output. This is not yet a verified AGAR raw-format parser.
- Implemented deterministic approximate count-bin stratification that keeps supplied groups intact, plus a group leakage validator. Missing group IDs default to image IDs and do not establish plate independence.
- Implemented manifest-derived EDA summaries and diagnostic plots for annotation count, category, lighting, resolution, and box size. It distinguishes zero-box images from source-labeled empty plates.
- `python -m compileall -q src`: passed. `python -m pytest -q`: **13 passed** in 1.34 s. `python -m colony_counting smoke`: passed. CLI help for `eda` and `data`: passed.
- These checks use synthetic fixtures only; they are software checks, not AGAR dataset evidence.
- Dataset-specific schema audit, split validation on actual plate groups, real EDA, classical inference, deep training, comparison, demo, and final project verification remain incomplete.

## 2026-10-21 continued

- Resolved the CUDA installation for the RTX 5050 in a dedicated Python 3.12.13 environment. CUDA 13.0 PyTorch completed an actual matrix multiplication; the observed system Python 3.14.5 remains CPU-only.
- Ran a one-epoch YOLO11n smoke on three generated disk images. Training, checkpoint reload, tiled prediction, and prediction overlay completed. This is a pipeline check only and provides no AGAR evidence.
- Implemented the local Gradio UI. `build_app` constructs a Blocks app, app callbacks return a classical overlay for an uploaded synthetic image and an explicit missing-checkpoint message for YOLO, and the live local server returned HTTP 200 on 127.0.0.1:7861. The server was stopped after the check.
- Exact environment lock now includes Gradio 6.30.0 and the resolved packages. Latest complete suite: 44 tests passed in both the GPU environment and the system Python 3.14.5 environment. Ruff passed in both.
- `python -m colony_counting verify` in `.venv-ml` reports `UNIT_VERIFIED`, with `project_verified=false`: the real data manifest and AGAR YOLO checkpoint are absent. No AGAR metrics exist.

## 2026-10-21 continued implementation and review

- Added a validation-only Optuna classical tuner. It requires the `val` partition, records a seed/search space/objective history, writes the selected YAML config and run metadata, and fingerprints manifest plus image bytes. Its one-trial fixture intentionally includes a nonexistent test image and confirms the test path is not read.
- Added split-specific batch prediction CLI for classical and YOLO. The YOLO checkpoint is loaded once per split; saved boxes retain category and confidence, with JSONL and a provenance sidecar. CLI tests verify split metadata and loader reuse.
- Added separate validation threshold selection for count MAE and detection F1, and confidence-ranked 101-point AP over IoU thresholds 0.50 through 0.95. AP's confidence-greedy matching is documented separately from fixed-threshold maximum-cardinality matching.
- Added a group-cluster paired bootstrap, populated count/metadata strata, Bland-Altman/count figures, traceable automatically ranked overlays, and paired saved-prediction ablation reporting.
- Added exact-image SHA-256 detection; byte-identical records are assigned to one split group without replacing source plate IDs. EDA reports duplicate file groups, and split validation checks duplicate hashes against partitions.
- Latest full checks: 58 tests pass in both `.venv-ml` (Python 3.12.13) and system Python 3.14.5; Ruff passes in both. `pip check` reports no broken requirements. Expanded project verification reports `UNIT_VERIFIED`, `project_verified=false` because real data and a real checkpoint are absent.
- After refactoring for one-load manifest inference, the saved synthetic YOLO checkpoint reloaded and completed tiled inference on the generated validation toy image. This is a software smoke only; its detections and runtime are not benchmark evidence.
- Final engineering checkpoint saved in `harness/FINAL_REPORT.md`. The workflow status is `BLOCKED` at AGAR source audit; downstream scientific milestones are explicitly blocked while the standalone local demo and software verification remain available.

## Continued implementation

- Added a configurable classical CV pipeline (`src/colony_counting/classical.py`), YAML defaults, per-stage diagnostics, and image CLI. The synthetic image integration test verifies toy-disk segmentation and output writing only.
- A first watershed test found a real code defect. Correcting watershed unknown-region initialization changed the toy fixture from zero detections to the expected three detections; this is a software debugging result, not a benchmark score.
- Added shared prediction JSONL, count metrics, configurable one-to-one IoU/centroid matching, paired image bootstrap (2,000 resamples by default), Wilcoxon handling, and an evaluation CLI. Known-answer tests cover undefined metrics, one-to-one maximum-cardinality assignment, serialization, seeds, ties, and all-zero differences.
- Ruff 0.17.1 installed. Latest full verification after code changes: Ruff check passed; `python -m pytest -q` reported **29 passed** in 1.54 s. Synthetic tests do not validate biological performance.
- Phase 0 remains open pending real authorized data. Classical model tuning, training, final evaluation, statistical claims, demo, and project verification remain incomplete.

