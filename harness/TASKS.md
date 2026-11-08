# Tasks

## In progress

- [x] Inspect workspace and record missing `PROJECT_SPEC.md`.
- [x] Audit OS, Python, storage/RAM, GPU, driver, torch and optional packages.
- [x] Add reconstructed requirements baseline and project navigation docs.
- [x] Implement generic COCO parser with path, dimension, and box validation.
- [x] Implement atomic JSONL manifest writer and data CLI commands.
- [x] Run parser/split/EDA tests (13 synthetic checks pass).
- [x] Implement deterministic grouped split utility and leakage validator (not audited against AGAR groups).
- [x] Hash exact image content, report duplicate groups, and keep byte-identical records in one split assignment group.
- [x] Implement manifest-derived EDA summary/figures; real AGAR EDA waits for authorized data.
- [ ] Confirm raw source format, group IDs, and integrity against an authorized AGAR package.
- [ ] Verify group leakage and EDA on actual data.
- [x] Implement configurable classical CV stages, diagnostics, YAML config, and CLI.
- [x] Implement model-independent prediction serialization, count/detection metrics, paired bootstrap, Wilcoxon, and evaluation CLI.
- [ ] Validate classical defaults and tuning on real training/validation data. (Tuning runner and synthetic isolation fixture are implemented.)
- [x] Add group-cluster bootstrap, fixed count/metadata strata, count error and Bland-Altman plots, and automatic ranked overlays.
- [x] Add paired saved-prediction ablation CLI with image and group bootstrap, Wilcoxon, split checks, and provenance note.
- [x] Add Optuna classical tuning runner restricted to the frozen validation split with config, seed, history, source fingerprint, and run metadata.
- [x] Add split-specific classical/YOLO batch predictions, preserving YOLO confidence and run provenance.
- [x] Add separate validation count-MAE/F1 threshold selector and confidence-ranked 101-point AP implementation.
- [ ] Generate real AGAR validation/test predictions, threshold selection, and mandatory scientific ablations after data is available.
- [x] Implement and locally serve the honest Gradio demo; app construction and missing-checkpoint handling verified.

## Pending phases

- [ ] Classical CV on real data and validation-only Optuna tuning.
- [ ] AGAR YOLO training/tiled inference, real validation threshold selection, and optional ONNX comparison.
- [ ] Real evaluation, group bootstrap, Wilcoxon interpretation, strata, mandatory ablations.
- [x] Reusable error analysis and gallery generation code; real-data gallery and human inspection remain pending predictions.
- [x] Local Gradio demo implementation and startup; model-backed YOLO UI still depends on AGAR training.
- [x] Write the final engineering checkpoint report and record all blockers without claiming project verification.
- [ ] Complete project acceptance after AGAR data and original protocol become available.

