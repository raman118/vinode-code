# Microbial colony counting

Research code for a reproducible classical CV vs. deep learning comparison on AGAR. The original project-specific specification was absent; see [PROJECT_SPEC.md](PROJECT_SPEC.md) for the requirements baseline reconstructed from the task prompt.

## Navigation

- Plan and phase acceptance checks: [harness/PLAN.md](harness/PLAN.md)
- Current machine-readable state: [harness/STATE.json](harness/STATE.json)
- Commands and recovery: [harness/RUNBOOK.md](harness/RUNBOOK.md)
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Experiment records: [docs/experiments.md](docs/experiments.md)
- Progress and known limits: [docs/progress.md](docs/progress.md), [docs/limitations.md](docs/limitations.md)
- Final checkpoint and blockers: [harness/FINAL_REPORT.md](harness/FINAL_REPORT.md)

## Setup and common commands

The verified GPU setup is isolated in `.venv-ml` on Python 3.12.13; its pinned Windows CUDA stack is recorded in `requirements-gpu-cu130.lock`. See [harness/RUNBOOK.md](harness/RUNBOOK.md) before recreating it. Run `python -m pytest`, `python -m ruff check src tests scripts`, and `python -m colony_counting verify` inside that environment. Dataset commands require an authorized local AGAR download.

## Engineering rules

- Never check dataset images, credentials, or unverified metrics into source.
- Do not use test data for tuning or model selection.
- Keep synthetic tests clearly separate from AGAR results.
- Record actual commands, exit codes, hardware, and artifacts in the harness.
- Do not mark work verified without passing its stated checks.

