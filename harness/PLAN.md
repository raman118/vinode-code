# Milestone plan

Statuses: `NOT_STARTED`, `IN_PROGRESS`, `BLOCKED`, `FAILED`, `VERIFIED`.

## M0: Environment and data foundation

- **Objective:** Build inspectable project foundation and validated data intake.
- **Prerequisites:** Python; authorized AGAR files for AGAR-specific checks.
- **Tasks:** Repo/docs/config/logging; source-schema adapter; manifest; group-aware splits; integrity checks; EDA and observed-data figures.
- **Artifacts:** package, config, dataset manifest/splits, EDA files, parser tests, runbook.
- **Validation:** `python -m pytest`; `python -m colony_counting validate-data ...`; split leakage check; EDA artifact review.
- **Completion:** AGAR schema confirmed from source files; records, dimensions, labels and boxes validated; split group leakage absent; EDA computed from real records.
- **Status:** BLOCKED. Generic parser, SHA-256 duplicate grouping, split, and EDA code are unit tested. No authorized AGAR data is available to confirm the raw schema, metadata, group IDs, or observed EDA.

## M1: Classical CV

- **Objective:** Implement configurable end-to-end classical colony counting.
- **Prerequisites:** Verified image data and annotations.
- **Tasks:** plate ROI/rim, illumination, channels/contrast, thresholds, morphology, distance markers/watershed, cluster correction, centroids/boxes/counts, diagnostics, Optuna validation tuning.
- **Artifacts:** YAML config, inference module, diagnostics, tuning records and predictions.
- **Validation:** component tests; inference on real images; diagnostic review; validation-only tuning provenance.
- **Completion:** pipeline executes on real AGAR images and produces validated predictions without test tuning.
- **Status:** BLOCKED. Configurable classical inference, split prediction, diagnostics, and validation-only Optuna runner are implemented and tested on synthetic fixtures. Real-image validation and Optuna tuning require the missing dataset.

## M2: YOLO training and inference

- **Objective:** Train and evaluate class-agnostic and feasible species-aware detectors.
- **Prerequisites:** validated splits; CUDA or documented CPU feasibility; compatible packages.
- **Tasks:** label conversion, sanity training, full training, checkpoint reload, tiled inference/deduplication, validation thresholds, optional ONNX comparison.
- **Artifacts:** labels, configs, experiment metadata/logs/checkpoints/predictions.
- **Validation:** geometry tests; train/val load; saved/reloaded model inference; tile merge checks.
- **Completion:** reproducible real-data training and inference paths with validation-selected thresholds.
- **Status:** BLOCKED. COCO-to-YOLO conversion, local-checkpoint training metadata, tiled/whole split inference, confidence-preserving predictions, separate validation MAE/F1 threshold selector, and geometry are implemented. One synthetic one-epoch train/reload/tiled-inference smoke passed on the RTX 5050. Real AGAR training, threshold selection, and ONNX comparison require the missing dataset.

## M3: Evaluation and statistics

- **Objective:** Evaluate saved predictions independently and reproducibly.
- **Prerequisites:** common ground-truth manifest and model predictions.
- **Tasks:** count/detection metrics, 2,000 paired bootstrap resamples, Wilcoxon edge cases/effect sizes, strata, tables/figures, mandatory ablations.
- **Artifacts:** reusable evaluator, result tables, plots, comparisons.
- **Validation:** known-answer metric fixtures; repeated evaluation consistency; identical image-set checks.
- **Completion:** all mandatory metrics/ablations generated from saved predictions and test untouched during selection.
- **Status:** BLOCKED. Core metrics, confidence-ranked AP, one-to-one matching, paired image/group bootstrap, Wilcoxon, JSONL I/O, fixed count/metadata strata, count plots/gallery, threshold selection, and saved-prediction ablation CLI pass synthetic tests. Actual prediction-based ablation runs, domain metrics, and real-data evaluation require AGAR predictions.

## M4: Error analysis

- **Objective:** Explain model errors with traceable figures and qualified findings.
- **Prerequisites:** final saved predictions and annotations.
- **Tasks:** overlays, worst-case gallery, Bland-Altman, strata, taxonomy, human inspection only when explicitly labeled.
- **Artifacts:** galleries, plots, analysis notes.
- **Validation:** output-to-prediction traceability and reviewed figures.
- **Completion:** analysis makes no unsupported causal or superiority claims.
- **Status:** BLOCKED. Reusable analysis writes count agreement and Bland-Altman plots, ranked saved-image overlays, and strata. Unit/integration tests pass. No AGAR predictions exist for a scientific gallery or manual review.

## M5: Local demo

- **Objective:** Provide an honest local Gradio demo using actual method artifacts.
- **Prerequisites:** working classical method; deep checkpoint for detector use.
- **Tasks:** upload, method selection, overlays/counts/latency/confidence, clear missing-model errors.
- **Artifacts:** app and startup docs.
- **Validation:** local startup and inference smoke check; missing checkpoint error check.
- **Completion:** functional local app with no mock inference or deployment claim.
- **Status:** IN_PROGRESS. Local Gradio app builds, classical callback and missing-checkpoint behavior pass, and localhost HTTP startup returned 200. Model-backed YOLO demo still awaits an AGAR-trained checkpoint.

## M6: Final verification and research review

- **Objective:** Reproduce and audit the complete project.
- **Prerequisites:** previous mandatory phases.
- **Tasks:** setup/import/tests/CLI/data/models/results/demo/docs/reviews; finalize README/report/resume bullets.
- **Artifacts:** `harness/FINAL_REPORT.md`, verified results, reproduction docs.
- **Validation:** project verification command and adversarial review; critical/high findings repaired.
- **Completion:** all mandatory acceptance criteria pass or remaining blockers are explicitly reported without claiming project verification.
- **Status:** BLOCKED. Code, CLI, unit, CUDA training/inference smoke, localhost demo checks, adversarial review, and final report are recorded. Mandatory real-data acceptance cannot pass without the dataset and original specification.

