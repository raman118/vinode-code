# Project requirements baseline

> The requested `PROJECT_SPEC.md` was not present in the workspace when work began. This file records the requirements visible in the supplied master engineering prompt; it is a reconstruction, not the missing original specification.

## Objective

Build a reproducible comparison of classical computer vision and deep object detection for microbial colony counting on the AGAR dataset. Report only measured results traceable to saved predictions and code.

## Data and experimental integrity

- Use official AGAR data only after access is actually available and license terms are respected. Record version, provenance, annotation structure, metadata, and limitations.
- Build a validated image/annotation manifest. Preserve empty, countable, and uncountable status where available; validate boxes and image references.
- Use deterministic group-aware 70/15/15 train/validation/test splits when feasible. Freeze splits before tuning. Keep test data out of all decisions.
- Required approximate planning cohort: 3,000 countable and 200 empty images, subject to verified availability.

## Methods

- Classical pipeline: plate/ROI detection, rim exclusion, illumination/background correction, contrast and channel selection, thresholding, morphology, distance transform, markers, watershed, cluster correction, centroids/boxes, and counts. Configure through YAML; save diagnostics; tune specified parameters with Optuna on training-derived validation data.
- Deep pipeline: class-agnostic YOLO detector and species-aware detector where feasible; verified label conversion; deterministic training; checkpoint and log preservation; overlapping tiled inference with global coordinates and deduplication; validation-selected count- and F1-oriented thresholds; ONNX export/agreement where supported.
- Evaluation: shared saved-prediction interface; counting and one-to-one detection metrics; fixed-seed paired bootstrap with 2,000 resamples; paired Wilcoxon tests with documented edge cases; stratified analyses and mandatory ablations.
- Analysis: real prediction-derived plots and worst-case galleries. Clearly separate observations, measurements, hypotheses, and limitations.
- Demo: local Gradio app with classical and deep inference, overlays, counts, latency, confidence control, and explicit missing-checkpoint errors.

## Required engineering and deliverables

Cross-platform setup, configuration, logging, parser, manifest/splits, data validation/EDA, modular methods, reusable evaluation, tests, CLI/task runner, reproducible experiment records, documentation, demo, and a final verification report. Required task commands include `setup`, `data`, `validate-data`, `eda`, `classical`, `tune-classical`, `train-yolo`, `infer-yolo`, `evaluate`, `analyze`, `ablations`, `smoke`, `test`, `lint`, `app`, and `verify`.

## Acceptance and reporting

Do not claim a phase verified until its declared checks pass. Distinguish static, unit, smoke, experiment, reproduced, and project verification. Synthetic fixtures may test code only and must never be described as AGAR evidence. Explicitly mark data-, GPU-, training-, or model-dependent work blocked when the prerequisite is absent. No fabricated sample sizes, metrics, timings, or experiment outcomes.

