# Method status

This document records implemented procedures and separates software checks from measured benchmark performance. Planned requirements are in [PROJECT_SPEC.md](../PROJECT_SPEC.md); acceptance states are tracked in [harness/PLAN.md](../harness/PLAN.md).

## Classical CV implementation status

`colony_counting.classical.count_colonies` implements channel selection, optional circular Hough plate detection with inscribed-circle fallback, rim exclusion, Gaussian background subtraction, CLAHE/percentile normalization, Otsu or adaptive thresholding, morphology, distance transform, local-maximum markers, watershed, area filters, optional component area correction, centroid/box extraction, and diagnostic stages. YAML defaults are in `configs/classical.yaml`.

The defaults are starting values, not trained or tuned AGAR settings. Pixel-scale parameters and channel/polarity choices need inspection and selection on real training/validation images. The code's synthetic disks establish that the stages can execute on a toy image; they do not establish colony-counting accuracy. No AGAR inference has been run.

## Evaluation implementation status

`colony_counting.evaluation` accepts JSONL records with per-image counts and optional boxes. It implements count metrics with explicit undefined values, configurable one-to-one IoU/centroid matching, confidence-ranked 101-point AP at IoU 0.50 and 0.50:0.05:0.95, separate validation selectors for count MAE and detection F1, paired image bootstrap CIs for MAE difference (2,000 resamples by default), and paired Wilcoxon tests with zero-difference and tie handling. `predict-classical` and `predict-yolo` emit common-schema split-specific prediction files with run metadata; the YOLO checkpoint is loaded once per split and confidence is preserved. The exact method-specific box matching contract from the missing original project spec is unknown; callers must provide the criterion and the current defaults are not represented as the original protocol.

Image bootstrap resamples images independently. Paired ablation reports also compute a group-cluster bootstrap when multiple supplied group IDs are available; all images within a sampled group travel together. Plate grouping still cannot be confirmed until source metadata is inspected. AP uses descending-confidence greedy one-to-one matching as in common object-detection evaluation, distinct from the maximum-cardinality assignment used for single-threshold precision/recall/F1.

## YOLO data and runtime status

`export_yolo_dataset` copies split images and writes normalized YOLO center-XYWH label files. It supports class-agnostic and category-aware labels and emits empty label files for zero-box images. `detection.tiling` handles edge-anchored overlapping grids, local-to-global XYXY conversion, confidence-ordered global NMS, and optional circular plate-interior filtering. `detection.yolo` contains an Ultralytics adapter, local-checkpoint training with configuration/source/dataset hashes and run metadata, and whole-image/tiled inference artifact output.

These paths completed a one-epoch integration smoke on generated toy images. The split prediction runner and confidence-preserving adapter are separately covered by CLI tests. No AGAR label conversion, training, checkpoint selection, or inference has occurred. Exact model-size and augmentation choices remain open pending the missing original spec and real data.

## Data split groundwork

`split_by_group` uses a seeded greedy allocation to approximate 70/15/15 image proportions and annotation-count strata while keeping each supplied `group_id` in one partition. Groups are indivisible, so exact ratios and strata are not guaranteed. The generic reader defaults missing group IDs to the image ID. Plate-level leakage cannot be ruled out until source metadata or duplicate analysis establishes a grouping strategy.

## EDA groundwork

`write_eda` summarizes only values present in validated records and emits JSON plus diagnostic figures. Zero-box records are not called empty plates without a source status label. No AGAR summary has been generated because no dataset is present.

