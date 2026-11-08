# Issues and blockers

## Open

### DATA-001: Missing project-specific requirements file and dataset

- **Class:** External/data-access blocker plus missing input artifact.
- **Evidence:** Workspace inspection found no files; recursive search found no `PROJECT_SPEC.md`; no AGAR images or annotations were present.
- **Impact:** Raw schema adapter, EDA, split integrity, real inference, model training, and scientific evaluation cannot be completed or claimed.
- **Action:** Reconstructed requirements from supplied prompt; did not submit the AGAR registration form. Resume dataset work when an authorized local package is available.

## Resolved

### GPU-001: System Python uses CPU-only PyTorch; resolved for project GPU work in isolated environment

- **Class:** Environment compatibility blocker.
- **Evidence:** The original Python 3.14.5 environment has `torch 2.12.0+cpu`. Created `.venv-ml` with Python 3.12.13, torch 2.14.0+cu130, torchvision 0.29.0+cu130, and OpenCV 4.14.0.94. RTX 5050 Laptop GPU reports compute capability 12.0; a CUDA matrix multiply passed.
- **Impact:** No remaining GPU environment blocker. Keep the CPU-only system interpreter distinct from the verified ML environment.
- **Action:** Use `.venv-ml` and `requirements-gpu-cu130.lock` for GPU experiments. No GPU issue remains.

## Open

### ML-001: Gradio demo has no AGAR-trained checkpoint

- **Evidence:** Gradio 6.30.0 local app builds and served HTTP 200 on 127.0.0.1; callback test verifies no mock inference when the checkpoint is missing. No AGAR-trained YOLO checkpoint exists.
- **Impact:** Classical demo is available. YOLO inference and scientifically relevant side-by-side comparison require a real trained model, which remains blocked by data availability.
- **Action:** Use `--checkpoint` only with an actual trained checkpoint and rerun app inference validation after AGAR training.

### EVAL-001: Scientific output generation is blocked by missing real data

- **Class:** External data-access blocker.
- **Evidence:** Analysis and ablation CLI paths pass synthetic integration tests, but the workspace contains no manifest, AGAR source files, saved AGAR predictions, or trained checkpoint.
- **Impact:** No real stratification, confidence selection, ablation result, statistical comparison, or error gallery can be produced.
- **Action:** On receipt of an authorized local package, verify its raw schema and grouping first; freeze split before tuning; then save validation and final test predictions before generating these reports.

### PROTOCOL-001: Missing original project specification

- **Class:** Missing project input.
- **Evidence:** No `PROJECT_SPEC.md` existed in the initially empty workspace. Current file reconstructs the supplied task requirements only.
- **Impact:** Any project-specific dataset paths, box-matching criteria, tolerance definitions, and other details omitted from the master prompt remain unverified.
- **Action:** Follow the explicit prompt-level requirements. Preserve protocol choices as configurable and do not claim conformance to unseen requirements.

