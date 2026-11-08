# Limitations and blockers

- The original project-specific `PROJECT_SPEC.md` was absent at initial inspection. The current one is reconstructed from the supplied prompt and may not preserve omitted project-specific details.
- No AGAR dataset files were present. The official site describes full access through registration under CC BY-NC 2.0 academic research terms. No registration or download was performed; raw schema, dataset version, groups, and local availability are unverified.
- The system Python 3.14.5 environment has PyTorch 2.12.0+cpu and cannot use CUDA. An isolated Python 3.12.13 environment has a verified CUDA stack (PyTorch 2.14.0+cu130, CUDA 13.0, RTX 5050 compute capability 12.0) and completed a real CUDA tensor operation.
- Ultralytics 8.4.183 is installed in the isolated GPU environment. A one-epoch training, checkpoint reload, tiled inference, and artifact smoke ran on generated synthetic toy images only. No AGAR training or inference has run.
- Gradio 6.30.0 local app is implemented and its UI callback and localhost HTTP startup have passed. It can run the classical pipeline and will require a trained YOLO checkpoint for deep inference. No trained AGAR checkpoint is available.
- The current COCO reader is generic and does not parse an uninspected AGAR release.
- Exact SHA-256 duplicates are detected and grouped during splitting. Perceptual duplicates from image transforms, re-encoding, or multiple views of one plate are not detected, and plate grouping remains unverified.
- No scientific results or conclusions have been measured.
- Analysis, group-bootstrap, and ablation commands are tested with synthetic saved predictions only. No AGAR stratified table, human-inspected gallery, or comparison has been generated.

