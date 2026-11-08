# Experiments

No AGAR experiment has run. There are no trained checkpoints, prediction artifacts, thresholds, measured metrics, or timing results.

The current parser checks operate on tiny synthetic image fixtures. They are unit tests, not scientific experiments and not benchmark evidence. The one-trial Optuna test also uses a synthetic validation image solely to test split isolation and artifact writing; its objective value is not a benchmark result.

## Software verification (not a scientific experiment)

- Classical toy-image pipeline and CLI execution were tested with generated disks. This verified execution, output shape, and diagnostics only.
- Evaluation functions were tested against known synthetic count/box examples, including one-to-one matching, undefined statistics, a fixed-seed 2,000-resample bootstrap, Wilcoxon ties/zero cases, and JSONL round trips.
- Current full suite reports 58 passing tests in the Python 3.12 CUDA environment and system Python 3.14 environment; Ruff passes. These values describe software validation only.

## Analysis and tuning code paths (synthetic software fixtures)

- The analysis CLI is integration-tested against a tiny saved-prediction file and manifest. It verifies frozen-split enforcement, count strata, metadata strata, count agreement and Bland-Altman figures, automatically ranked overlays, and paired baseline/variant reports.
- The Optuna CLI has a one-trial fixture with one validation image and an intentionally nonexistent test image. The trial completed and wrote best YAML, CSV history, and experiment metadata without loading the test path. This verifies code behavior only; the recorded fixture objective is not reported as research evidence.
- Group bootstrap resamples supplied plate-group IDs while retaining all images within each selected group. Actual plate grouping is unknown until the source dataset is inspected.
- Validation threshold selection was tested on saved prediction fixtures carrying confidence values. Its output separately minimizes count MAE and maximizes one-to-one detection F1; no AGAR thresholds have been selected.
- Confidence-ranked class-aware AP calculation was checked on known toy boxes. It does not constitute the requested YOLO mAP result because no AGAR checkpoint or predictions were evaluated.
- Manifest-wide prediction runners were integration-tested on a synthetic classical image and with a fake YOLO loader to verify checkpoint-load reuse and confidence serialization. Actual post-refactor tiled inference was also run on the saved synthetic YOLO checkpoint.
- Exact-file duplicate hashing and split co-assignment were tested with two synthetic image records carrying different plate IDs but identical content hashes. No AGAR duplicate rate is measured.

## Synthetic YOLO integration smoke (not AGAR evidence)

- **ID:** `synthetic-smoke-20261021-014702`
- **Status:** Completed, one training epoch. Data are 3 generated 96x96 toy images with one drawn disk each; random YOLO11n initialization, seed 42, image size 64, batch 2, AMP enabled, workers 0.
- **Hardware/software:** Windows 11, NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB reported), Python 3.12.13, torch 2.14.0+cu130, Ultralytics 8.4.183.
- **Artifacts:** `results/synthetic-yolo-smoke-20261021-014702/`, including generated train/validation files, experiment metadata, Ultralytics CSV/log/plots, saved checkpoint, tiled prediction and overlay.
- **Purpose:** Exercise YOLO dataset loading, forward/backward training, AMP check, checkpoint write/reload, tiled inference, and output serialization. This run has no bearing on AGAR performance and is excluded from scientific comparisons.

