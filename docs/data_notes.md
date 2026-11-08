# Data notes

## Workspace observation

No AGAR files, labels, checkpoints, or existing data artifacts were present in the inspected workspace. The AGAR image count, accessible sample size, exact annotation schema, category IDs, lighting/camera metadata, duplicate identities, and plate groups are therefore **NOT MEASURED** here.

## Official source and access

The [official AGAR page](https://agar.neurosys.com/) describes 18,000 images and 336,442 labeled colonies in its dataset overview, states a CC BY-NC 2.0 academic research license, and presents a registration form for full research access. These are statements on the source page, not a local verification of accessible files or release contents. No registration was submitted and no data were downloaded.

## Compute environment update

After checking the current official PyTorch release matrix and NVIDIA's compute capability table, an isolated Python 3.12.13 environment was created. The RTX 5050 Laptop GPU reports compute capability 12.0; PyTorch 2.14.0+cu130 reports CUDA available and completed a 1024x1024 GPU matrix operation. The prior Python 3.14.5 environment still has CPU-only torch. Resolved isolated package versions are in `requirements-gpu-cu130.lock`.

## Current software assumption

`src/colony_counting/data/coco.py` accepts a COCO-style JSON interchange format. It checks every referenced image exists and decodes, dimensions match, categories and IDs are consistent, and every XYWH box is finite, positive, and within bounds. Optional `group_id`, `lighting`, `camera`, and `status` image keys are retained. Missing groups default to each individual image ID, which avoids claiming any unobserved relationship.

This format is not asserted to match raw AGAR. Once an authorized sample is supplied, inspect its format and provenance first, then implement and test a source-specific adapter.

The generic parser also records SHA-256 hashes for exact-file duplicate detection. Split assignment joins records that share a supplied group ID or byte-identical image content, while retaining original group IDs for later cluster statistics. This catches exact duplicates only; resized, recompressed, or otherwise near-duplicate plates need additional image-level review.

