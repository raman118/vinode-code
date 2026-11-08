# Build log

## 2026-10-21, initial inspection

- Command: `Get-Location; Get-ChildItem -Force; if (Test-Path PROJECT_SPEC.md) { Get-Content PROJECT_SPEC.md -Raw } else { 'PROJECT_SPEC.md MISSING' }; git status --short --branch`
- Result: workspace path was `C:\Users\rcho3\Desktop\vinod-code`; no project files were present; `PROJECT_SPEC.md` missing.
- Command: Python/system audit using Python `platform`, `shutil`, `psutil`, `nvidia-smi`, and torch import.
- Result: Windows 11, Python 3.14.5, 108,256,247,808 free bytes, 25,380,823,040 total RAM bytes. `nvidia-smi`: RTX 5050, 8,151 MiB, driver 592.82, CUDA compatibility 13.1. PyTorch 2.12.0+cpu; CUDA unavailable.
- Package probe: NumPy, OpenCV, Pillow, pandas, SciPy, Matplotlib, PyYAML, pytest, torch, and Optuna available; Ultralytics and Gradio unavailable.
- Official source reviewed: <https://agar.neurosys.com/>. Page states CC BY-NC 2.0 academic research access and shows full-dataset registration. No form submitted and no dataset downloaded.
- Files created: requirements baseline, navigation/architecture docs, package skeleton, COCO parser, CLI, synthetic tests, harness state and plan.
- Command: `python -m pip install -e . --no-deps`
- Result: editable package built and installed successfully (`microbial-colony-counting 0.1.0`).
- Commands: `python -m compileall -q src`; `python -m pytest -q`; `python -m colony_counting smoke`; `python -m colony_counting eda --help`; `python -m colony_counting data --help`.
- Results: compileall passed; **13 passed** in 1.34 s; smoke import passed; data/EDA CLI help parsed. Tests are synthetic software fixtures only; no AGAR observations or metrics were generated.
- Classical implementation: configurable CV pipeline, YAML config, diagnostics, and CLI added. First run `python -m pytest -q tests/test_classical.py` failed because watershed left marker pixels only. Root cause: all binary interior pixels were assigned background. Initialization corrected to unknown foreground with seeded background/peaks; classical tests then passed (4 passed).
- Evaluation implementation: saved prediction JSONL, count metrics, one-to-one IoU/centroid matcher, paired MAE bootstrap, Wilcoxon, and CLI added. First full integration run exposed `UnboundLocalError` for local `json` import in evaluation CLI; import moved to module scope.
- Ruff 0.17.1 was installed to run configured lint checks. `python -m ruff format src tests`: 15 files unchanged on final run. `python -m ruff check src tests`: All checks passed. `python -m pytest -q`: **29 passed** in 1.54 s.
- All current tests use toy images/records. They do not constitute an AGAR experiment, trained model, or performance result.

## 2026-10-21, model evaluation and split integrity additions

- Added validation threshold selection for count MAE and detection F1, requiring confidence on predicted boxes and a manifest whose prediction images all belong to `val`. Added confidence-ranked 101-point AP at IoU 0.50 and 0.50 through 0.95. Tests use known toy boxes only.
- Added `predict-classical` and `predict-yolo` split runners. YOLO loads a checkpoint once for a manifest split and preserves confidence/category in prediction JSONL. Run sidecars capture split, manifest/checkpoint hash, source fingerprint, runtime, and hardware.
- First YOLO batch runner integration test failed because the inference helper did not accept the class-agnostic NMS argument. Corrected helper interface. The test initially patched the leaf module instead of the function's imported module boundary; changed the test seam to patch `manifest_inference`. Full tests then passed.
- Added SHA-256 for exact file identity during normalized COCO parsing. Split assignment unions explicit plate groups and exact duplicate image content, while preserving original `group_id`; EDA reports exact duplicate groups. Unit fixture verifies identical content with separate group labels remains in one split.
- Commands: `.venv-ml\\Scripts\\ruff.exe format src tests scripts`, `.venv-ml\\Scripts\\ruff.exe check src tests scripts`, `.venv-ml\\Scripts\\python.exe -m pytest -q`.
- Results: Ruff clean; **58 passed** in 4.06 s under Python 3.12.13 GPU env; same suite **58 passed** in 3.92 s under system Python 3.14.5; Ruff clean under both.
- Commands: `.venv-ml\\Scripts\\python.exe -m colony_counting verify`; `.venv-ml\\Scripts\\python.exe -m pip check`.
- Results: project verifier `UNIT_VERIFIED`, `project_verified=false`; manifest and AGAR checkpoint are reported blocked. `pip check` reports no broken requirements.
- Refactored inference smoke: `.venv-ml\\Scripts\\python.exe -m colony_counting infer-yolo --image results/synthetic-yolo-smoke-20261021-014702/dataset/images/val/toy-0.png --weights results/synthetic-yolo-smoke-20261021-014702/training/synthetic-smoke-20261021-014702/train/weights/best.pt --output results/synthetic-yolo-smoke-20261021-014702/refactored-inference --tile-size 64 --overlap 0.2 --image-size 64 --confidence 0.001 --device 0`.
- Result: loaded saved synthetic checkpoint and wrote prediction/overlay after tiled inference. This smoke does not measure AGAR performance.
- Gradio checks: `build_app` returned a `Blocks` instance. `python -m colony_counting app --config configs/classical.yaml --server-name 127.0.0.1 --server-port 7861` was queried with `Invoke-WebRequest`; response was HTTP 200 and the process was stopped. App starts loopback-only and `share=False`.
- No AGAR dataset or annotations were made available. No AGAR training, inference, tuning, mAP, threshold selection, ablation, or scientific comparison occurred.

