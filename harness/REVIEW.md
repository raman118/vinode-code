# Review log

## Initial independent review perspectives (2026-10-21)

### Software engineering

- Initial COCO reader includes schema errors, duplicate IDs, file existence/decode, dimension parity, bounds, and atomic manifest output. Parser checks still need execution.
- Generic parser does not confirm that real AGAR uses COCO. Source-specific contract remains unverified.
- **Severity MEDIUM:** CLI includes future tasks that return an explicit not-implemented status. Keep these clearly labeled; no stub may imply the behavior exists.

### Computer vision

- No image-processing method exists yet, so ROI, watershed, morphology, and scale behavior are not reviewed or measured.
- Box checks use annotated image dimensions and now compare them to decoded dimensions.

### ML research

- No data splits or model selection have occurred. Leakage cannot yet be assessed beyond absent data.
- Official access and license terms must be recorded before data use. Test-set isolation remains a hard requirement.

### Adversarial quality

- The top-level spec is a reconstructed prompt baseline, not the original project spec.
- Website overview statistics are source-page statements only; they must not be represented as locally accessible data counts.
- Synthetic parser tests can prove code behavior but not AGAR compatibility or counting quality.
- No reported scientific numbers currently exist.

## Follow-up review after GPU, synthetic YOLO, and demo integration

- **MEDIUM, fixed:** The first Gradio integration was not yet installed or build-tested. Gradio 6.30.0 is now in the resolved lock; Blocks construction, callback behavior, and local HTTP startup were exercised. UI startup binds only to loopback by default and sets `share=False`.
- **MEDIUM, open:** Review of saved AGAR data and model-backed demo remains impossible until authorized AGAR data and a trained checkpoint exist. The app's missing-checkpoint status is explicit and does not fabricate detections.
- **MEDIUM, open:** No stratified score tables, prediction-based ablation reports, human-inspected failure gallery, species-aware detector, ONNX consistency comparison, or AGAR test evaluation exists. Do not infer those results from the synthetic smoke.
- **LOW, open:** Test coverage currently proves code paths and known-answer synthetic cases; it does not establish system performance for biological plate images.

## Follow-up review after stratified analysis and validation tuning

- **MEDIUM, fixed:** Analysis now requires all saved prediction IDs to exist in the manifest and to match the explicitly selected split. The CLI reports missing metadata and wrong-split predictions as errors rather than silently combining data.
- **MEDIUM, fixed:** Group bootstrap uses the manifest's supplied group identifiers and resamples whole groups. When only one group is present, the paired ablation command reports the group interval as unavailable rather than pretending image independence is resolved.
- **MEDIUM, fixed:** Tuning reads only rows assigned to `val`; the test fixture includes a nonexistent test image to check the filter. It records the manifest and image fingerprint, search settings, seed, trial count, history, and output config. Optuna scores are validation selection metrics, not final results.
- **HIGH, open:** The original `PROJECT_SPEC.md` is unavailable. The chosen count strata and COCO normalization contract may not match requirements omitted from the supplied prompt. They remain documented choices, not verified original protocol.
- **HIGH, open:** No AGAR dataset package or authoritative plate-group IDs are present. This prevents assessing source semantics, leakage, lighting/camera strata, the inference code on biological textures, or any scientific claim.
- **MEDIUM, fixed:** Exact byte duplicates are hashed during normalized data ingestion and are grouped together during split assignment. This does not detect perceptual duplicates such as crops, rotations, or re-encodings; inspect those if AGAR plate relationships are available.

### Follow-up review after first implementations

- **MEDIUM, fixed:** The initial watershed initialization fixed foreground pixels as background, so only marker pixels were counted. A three-disk test exposed this. Foreground is now initialized as unknown and seeded by distance peaks; synthetic test passes.
- **MEDIUM, fixed:** The first evaluation CLI test found `json` imported locally after a prior reference, causing `UnboundLocalError`. Import was moved to module scope; end-to-end CLI test passes.
- **MEDIUM, fixed:** Ruff review found an unused cluster-suppression object and weak finite-value logic. Removed the unused object and use `math.isfinite`; input validation was tightened.
- **MEDIUM, open:** Classical defaults are not scientifically justified until inspected and tuned on AGAR validation data. Synthetic checks do not address illumination, colony morphology, dense plates, plate boundary failure, or texture artifacts.
- **MEDIUM, open:** Image bootstrap assumes independent images. Plate-group dependence needs group bootstrap or an explicit documented limitation before inferential claims.
- **MEDIUM, fixed:** Confidence-preserving batch YOLO predictions, separate validation count-MAE/F1 threshold selection, and confidence-ranked AP are now implemented and tested on known toy predictions. No AGAR mAP or threshold values are measured.

