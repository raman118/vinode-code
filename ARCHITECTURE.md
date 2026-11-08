# Architecture

The code is organized as an installable Python package with thin CLI commands and independently testable components.

- `colony_counting.data`: normalized COCO-style annotation parsing, validation, manifests, and deterministic group splits.
- `colony_counting.classical`: configured image-processing pipeline and diagnostics.
- `colony_counting.detection`: YOLO label conversion, training/inference adapters, and tile geometry.
- `colony_counting.manifest_inference`: frozen-split classical and YOLO prediction runners.
- `colony_counting.evaluation`: shared prediction schema, count/detection metrics, confidence threshold selection, AP, bootstrap intervals, and paired tests.
- `colony_counting.tuning`: Optuna validation-only parameter search with experiment provenance.
- `colony_counting.analysis`: plots, stratification, and failure galleries derived from saved predictions.
- `colony_counting.app`: local Gradio entry point.
- `configs/`: versioned YAML settings. `data/` and `results/` are runtime-only and are not source evidence until their contents are generated and validated.

The AGAR-specific raw schema adapter remains unverified until an authorized sample or full dataset is available. Generic COCO ingestion is an initial interchange format, not a claim about the AGAR release schema. YOLO integration has only been smoke-tested on generated toy data.

