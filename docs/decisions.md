# Decisions

## Requirements baseline when the requested spec is missing

**Decision:** Preserve `PROJECT_SPEC.md` as an explicit reconstruction of the supplied master prompt and state that it is not the missing original.

**Reason:** The task permits using requirements in the prompt when the file is absent, but scientific fidelity requires disclosing the missing contract.

## Generic COCO input as a first adapter

**Decision:** Implement an explicit generic COCO reader as an initial normalized input path; do not label it an AGAR adapter.

**Reason:** The workspace contains no dataset or annotation sample from which AGAR's exact schema could be verified. Generic validation enables real software progress without inventing schema details.

## No dataset registration on the user's behalf

**Decision:** Do not submit the official dataset access form.

**Reason:** The official site requires registration and collection of personal/contact details. No such details were provided. Continue independent code work; real-data work needs an authorized local download.

## Python environment

**Decision:** Keep the pre-existing system Python 3.14.5/CPU-only torch environment untouched and create an isolated Python 3.12.13 GPU environment using CUDA 13.0 wheels.

**Reason:** The NVIDIA RTX 5050 reports compute capability 12.0. Current PyTorch release guidance supports Blackwell on CUDA 13.0+ and the installed driver is sufficient. A CUDA tensor operation verified the isolated `torch 2.14.0+cu130` build. Exact resolved package versions are recorded in `requirements-gpu-cu130.lock`.

## Exact duplicate images are split together

**Decision:** Hash image bytes while validating the normalized manifest. During split creation, union byte-identical image records with each other and with all records sharing their supplied group ID. Preserve original `group_id` separately from the derived `split_group_id`.

**Reason:** Two identical files with different IDs or plate labels can otherwise cross train/test partitions. Hash equality detects exact duplicates without making unverified claims about perceptual similarity. It does not replace manual near-duplicate inspection.

