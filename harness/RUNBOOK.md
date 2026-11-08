# Runbook

## Environment

The observed working CUDA stack used the separately installed CPython 3.12.13 at `C:\Users\rcho3\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe`, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, NVIDIA driver 592.82, and the RTX 5050 Laptop GPU. A tested environment lock is provided for Windows x64 in `requirements-gpu-cu130.lock`. The 3.14 system interpreter has CPU-only PyTorch and is not the verified GPU environment.

On Windows PowerShell, the reproducible environment setup is:

```powershell
& 'C:\Users\rcho3\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe' -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install -r requirements-gpu-cu130.lock
.\.venv-ml\Scripts\python.exe -m pip install -e . --no-deps
```

To set up another environment, install the optional extras only as needed: `.[analysis]`, `.[deep]`, `.[app]`. Verify the installed PyTorch CUDA wheel against Python, GPU capability, and driver, then run an actual CUDA tensor operation before training.

The verified lock includes Gradio 6.30.0. Launch the local demo with `.\.venv-ml\Scripts\python.exe -m colony_counting app --config configs\classical.yaml`. Add `--checkpoint path\to\best.pt` only for a real trained model. The app binds to 127.0.0.1 and does not create a public share link.

## Current checks

```powershell
python -m pytest
python -m colony_counting smoke
python -m colony_counting verify
python -m colony_counting app --config configs\classical.yaml
python -m colony_counting validate-data --annotations path\to\annotations.json --images path\to\images
python -m colony_counting data --annotations path\to\annotations.json --images path\to\images --output data\processed\manifest.jsonl
python -m colony_counting prepare-yolo --annotations path\to\annotations.json --images path\to\images --output data\yolo\colony --seed 42
python -m colony_counting tune-classical --manifest data\processed\manifest.jsonl --images-root path\to\images --config configs\classical.yaml --trials 50 --seed 42 --output results\classical-tuning
python -m colony_counting analyze --predictions results\predictions.jsonl --manifest data\processed\manifest.jsonl --split test --images-root path\to\images --output results\analysis
python -m colony_counting ablations --baseline results\predictions\baseline.jsonl --variant tuned=results\predictions\tuned.jsonl --manifest data\processed\manifest.jsonl --split test --output results\ablations.json
python -m colony_counting predict-classical --manifest data\processed\manifest.jsonl --images-root path\to\images --split val --config configs\classical.yaml --output results\predictions\classical-val.jsonl
python -m colony_counting predict-yolo --manifest data\processed\manifest.jsonl --images-root path\to\images --split val --checkpoint artifacts\yolo\best.pt --confidence-floor 0.001 --output results\predictions\yolo-val.jsonl
python -m colony_counting tune-thresholds --predictions results\predictions\yolo-val.jsonl --manifest data\processed\manifest.jsonl --output results\yolo-thresholds.json
```

Synthetic YOLO integration smoke (one epoch on generated toy data, never benchmark evidence):

```powershell
.\.venv-ml\Scripts\python.exe scripts\synthetic_yolo_smoke.py
```

The current parser expects COCO JSON as a normalized input format. Do not use it as an AGAR parser until an authorized sample confirms the source schema.

The tuning command hard-requires the `val` split and writes a search config, CSV history, and run metadata. Analysis and ablation commands only summarize already-saved prediction files; their split argument must match each prediction image's frozen manifest split. Run model selection and confidence selection on validation predictions before producing final test predictions.

## Resume procedure

Read `AGENTS.md`, `harness/STATE.json`, `harness/PLAN.md`, and recent `harness/build-log.md` entries. Inspect files and active processes, reconcile with recorded state, then work the first unfinished task in `harness/TASKS.md`.

