"""Cross-platform command-line entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from colony_counting.data.agar import load_agar_records
from colony_counting.data.coco import DatasetFormatError, load_coco_records, write_manifest
from colony_counting.data.splits import split_by_group


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="colony-counting", description="Colony counting research tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="Build a validated manifest from COCO JSON")
    data_parser.add_argument("--annotations", type=Path, required=True)
    data_parser.add_argument("--images", type=Path, required=True)
    data_parser.add_argument("--output", type=Path, default=Path("data/processed/manifest.jsonl"))
    data_parser.add_argument("--seed", type=int, default=42)
    data_parser.add_argument(
        "--no-split", action="store_true", help="Write records without train/val/test assignment"
    )

    agar_data_parser = subparsers.add_parser(
        "agar-data",
        help="Build a validated manifest directly from the AGAR per-image JSON layout",
    )
    agar_data_parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing countable/, empty/, uncountable/ sub-dirs",
    )
    agar_data_parser.add_argument(
        "--output", type=Path, default=Path("data/processed/manifest.jsonl")
    )
    agar_data_parser.add_argument("--seed", type=int, default=42)
    agar_data_parser.add_argument(
        "--no-split", action="store_true", help="Write records without train/val/test assignment"
    )

    validate_parser = subparsers.add_parser(
        "validate-data", help="Validate annotations and referenced images"
    )
    validate_parser.add_argument("--annotations", type=Path, required=True)
    validate_parser.add_argument("--images", type=Path, required=True)

    eda_parser = subparsers.add_parser(
        "eda", help="Summarize validated data and write diagnostic plots"
    )
    eda_parser.add_argument("--annotations", type=Path, required=True)
    eda_parser.add_argument("--images", type=Path, required=True)
    eda_parser.add_argument("--output", type=Path, default=Path("results/eda"))

    classical_parser = subparsers.add_parser(
        "classical", help="Count colonies in one image using classical CV"
    )
    classical_parser.add_argument("--image", type=Path, required=True)
    classical_parser.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    classical_parser.add_argument("--output", type=Path, default=Path("results/classical"))

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate saved method-independent predictions"
    )
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    evaluate_parser.add_argument("--match-mode", choices=("iou", "centroid"), required=True)
    evaluate_parser.add_argument("--match-threshold", type=float, required=True)
    evaluate_parser.add_argument("--require-category-match", action="store_true")
    evaluate_parser.add_argument("--absolute-tolerance", type=int, default=5)
    evaluate_parser.add_argument("--relative-tolerance", type=float, default=0.10)
    evaluate_parser.add_argument("--compare", type=Path)
    evaluate_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser.add_argument(
        "--map-class-aware",
        action="store_true",
        help="Compute species/category AP instead of class-agnostic AP",
    )

    analyze_parser = subparsers.add_parser(
        "analyze", help="Write count strata, count plots, and optional worst-case overlays"
    )
    analyze_parser.add_argument("--predictions", type=Path, required=True)
    analyze_parser.add_argument("--manifest", type=Path, required=True)
    analyze_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    analyze_parser.add_argument("--images-root", type=Path)
    analyze_parser.add_argument("--output", type=Path, default=Path("results/error-analysis"))
    analyze_parser.add_argument("--gallery-size", type=int, default=12)

    ablation_parser = subparsers.add_parser(
        "ablations", help="Compare saved paired prediction files without retraining"
    )
    ablation_parser.add_argument("--baseline", type=Path, required=True)
    ablation_parser.add_argument("--variant", action="append", required=True, metavar="NAME=PATH")
    ablation_parser.add_argument("--manifest", type=Path, required=True)
    ablation_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    ablation_parser.add_argument("--output", type=Path, default=Path("results/ablations.json"))
    ablation_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    ablation_parser.add_argument("--seed", type=int, default=42)

    tune_parser = subparsers.add_parser(
        "tune-classical", help="Tune classical settings on a frozen validation manifest split"
    )
    tune_parser.add_argument("--manifest", type=Path, required=True)
    tune_parser.add_argument("--images-root", type=Path, required=True)
    tune_parser.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    tune_parser.add_argument("--output", type=Path, default=Path("results/classical-tuning"))
    tune_parser.add_argument("--trials", type=int, default=50)
    tune_parser.add_argument("--seed", type=int, default=42)

    threshold_parser = subparsers.add_parser(
        "tune-thresholds", help="Choose separate MAE and detection-F1 thresholds on validation data"
    )
    threshold_parser.add_argument("--predictions", type=Path, required=True)
    threshold_parser.add_argument("--manifest", type=Path, required=True)
    threshold_parser.add_argument(
        "--output", type=Path, default=Path("results/yolo-thresholds.json")
    )
    threshold_parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[i / 100 for i in range(5, 100, 5)]
    )
    threshold_parser.add_argument("--match-mode", choices=("iou", "centroid"), default="iou")
    threshold_parser.add_argument("--match-threshold", type=float, default=0.5)
    threshold_parser.add_argument("--require-category-match", action="store_true")

    classical_batch_parser = subparsers.add_parser(
        "predict-classical", help="Generate saved predictions for one frozen manifest split"
    )
    classical_batch_parser.add_argument("--manifest", type=Path, required=True)
    classical_batch_parser.add_argument("--images-root", type=Path, required=True)
    classical_batch_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    classical_batch_parser.add_argument(
        "--config", type=Path, default=Path("configs/classical.yaml")
    )
    classical_batch_parser.add_argument("--output", type=Path, required=True)

    yolo_batch_parser = subparsers.add_parser(
        "predict-yolo", help="Generate scored tiled or whole-image predictions for one split"
    )
    yolo_batch_parser.add_argument("--manifest", type=Path, required=True)
    yolo_batch_parser.add_argument("--images-root", type=Path, required=True)
    yolo_batch_parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    yolo_batch_parser.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    yolo_batch_parser.add_argument("--checkpoint", type=Path, required=True)
    yolo_batch_parser.add_argument("--output", type=Path, required=True)
    yolo_batch_parser.add_argument("--confidence-floor", type=float, default=0.001)
    yolo_batch_parser.add_argument("--image-size", type=int, default=640)
    yolo_batch_parser.add_argument("--tile-size", type=int, default=1024)
    yolo_batch_parser.add_argument("--overlap", type=float, default=0.2)
    yolo_batch_parser.add_argument("--nms-iou", type=float, default=0.5)
    yolo_batch_parser.add_argument("--device", default="auto")
    yolo_batch_parser.add_argument("--whole-image", action="store_true")
    yolo_batch_parser.add_argument("--class-aware-nms", action="store_true")

    prepare_yolo_parser = subparsers.add_parser(
        "prepare-yolo", help="Export a frozen manifest into YOLO image and label folders"
    )
    prepare_yolo_parser.add_argument("--annotations", type=Path, required=True)
    prepare_yolo_parser.add_argument("--images", type=Path, required=True)
    prepare_yolo_parser.add_argument("--output", type=Path, required=True)
    prepare_yolo_parser.add_argument("--seed", type=int, default=42)
    prepare_yolo_parser.add_argument("--species-aware", action="store_true")

    agar_prepare_yolo_parser = subparsers.add_parser(
        "agar-prepare-yolo",
        help="Export AGAR manifest directly into YOLO image and label folders",
    )
    agar_prepare_yolo_parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.jsonl"),
        help="Frozen split manifest built by agar-data"
    )
    agar_prepare_yolo_parser.add_argument(
        "--images-root", type=Path, required=True,
        help="Root that, joined with manifest image_path, gives the absolute image path"
    )
    agar_prepare_yolo_parser.add_argument("--output", type=Path, required=True)
    agar_prepare_yolo_parser.add_argument(
        "--species-aware", action="store_true",
        help="Use per-species class ids instead of class-agnostic 'colony'"
    )
    agar_prepare_yolo_parser.add_argument(
        "--countable-only", action="store_true", default=True,
        help="Only export countable images (default: True)"
    )

    train_parser = subparsers.add_parser(
        "train-yolo", help="Train YOLO from a local checkpoint and dataset"
    )
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--weights", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, default=Path("results/yolo"))
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--image-size", type=int, default=640)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--patience", type=int, default=20)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--name", default="colony-yolo")

    infer_parser = subparsers.add_parser("infer-yolo", help="Run local YOLO checkpoint inference")
    infer_parser.add_argument("--image", type=Path, required=True)
    infer_parser.add_argument("--weights", type=Path, required=True)
    infer_parser.add_argument("--output", type=Path, default=Path("results/yolo/inference"))
    infer_parser.add_argument("--tile-size", type=int, default=1024)
    infer_parser.add_argument("--overlap", type=float, default=0.2)
    infer_parser.add_argument("--image-size", type=int, default=640)
    infer_parser.add_argument("--confidence", type=float, default=0.25)
    infer_parser.add_argument("--nms-iou", type=float, default=0.5)
    infer_parser.add_argument("--device", default="auto")
    infer_parser.add_argument("--whole-image", action="store_true")
    infer_parser.add_argument("--class-aware-nms", action="store_true")
    infer_parser.add_argument("--plate-circle", type=float, nargs=3, metavar=("X", "Y", "R"))
    infer_parser.add_argument("--plate-roi-scale", type=float, default=0.92)

    app_parser = subparsers.add_parser("app", help="Launch the local Gradio colony-counting demo")
    app_parser.add_argument("--config", type=Path, default=Path("configs/classical.yaml"))
    app_parser.add_argument("--checkpoint", type=Path)
    app_parser.add_argument("--server-name", default="127.0.0.1")
    app_parser.add_argument("--server-port", type=int, default=7860)

    verify_parser = subparsers.add_parser(
        "verify", help="Run code gates and report project prerequisites"
    )
    verify_parser.add_argument("--output", type=Path, default=Path("results/verification.json"))

    for command, help_text in (
        ("setup", "Show environment setup guidance"),
        ("smoke", "Check core imports and configuration"),
        ("test", "Run the test suite"),
        ("lint", "Run static lint checks"),
    ):
        subparsers.add_parser(command, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a project task and return a process status."""

    args = build_parser().parse_args(argv)
    try:
        if args.command in {"data", "validate-data", "eda"}:
            records = load_coco_records(args.annotations, args.images)
            if args.command == "data":
                if not args.no_split:
                    records = split_by_group(records, seed=args.seed)
                write_manifest(records, args.output)
                print(
                    f"Validated {len(records)} records; "
                    f"split={'disabled' if args.no_split else f'seeded (seed={args.seed})'}; "
                    f"manifest written to {args.output}"
                )
            elif args.command == "eda":
                from colony_counting.data.eda import write_eda

                summary = write_eda(records, args.output)
                print(
                    f"EDA written to {args.output}; "
                    f"{summary['image_count']} images and {summary['annotated_object_count']} annotations summarized"
                )
            else:
                print(f"Validated {len(records)} records from {args.annotations}")
            return 0
        if args.command == "agar-data":
            records = load_agar_records(args.data_root)
            if not args.no_split:
                records = split_by_group(records, seed=args.seed)
            write_manifest(records, args.output)
            countable = sum(1 for r in records if r.status == "countable")
            empty = sum(1 for r in records if r.status == "empty")
            uncountable = sum(1 for r in records if r.status == "uncountable")
            print(
                f"AGAR manifest: {len(records)} total records "
                f"({countable} countable, {empty} empty, {uncountable} uncountable); "
                f"split={'disabled' if args.no_split else f'seeded (seed={args.seed})'}; "
                f"written to {args.output}"
            )
            return 0
        if args.command == "prepare-yolo":
            from colony_counting.detection.yolo_data import export_yolo_dataset

            records = split_by_group(
                load_coco_records(args.annotations, args.images), seed=args.seed
            )
            manifest_path = args.output / "manifest.jsonl"
            write_manifest(records, manifest_path)
            dataset_yaml = export_yolo_dataset(
                records,
                args.images,
                args.output,
                class_agnostic=not args.species_aware,
            )
            print(f"YOLO data exported; manifest: {manifest_path}; dataset YAML: {dataset_yaml}")
            return 0
        if args.command == "classical":
            import cv2

            from colony_counting.classical import (
                count_colonies,
                load_classical_config,
                save_diagnostics,
            )

            image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Cannot decode image: {args.image}", file=sys.stderr)
                return 1
            config = load_classical_config(args.config)
            result = count_colonies(image, config)
            args.output.mkdir(parents=True, exist_ok=True)
            diagnostics = save_diagnostics(result, args.output / "diagnostics")
            payload = {
                "image": str(args.image),
                "count": result.count,
                "plate_circle": result.plate_circle,
                "detections": [
                    {
                        "center_x": item.x,
                        "center_y": item.y,
                        "bbox_xywh": item.box,
                        "area": item.area,
                        "source": item.source,
                    }
                    for item in result.detections
                ],
                "config": config.__dict__,
                "diagnostic_files": [str(path) for path in diagnostics],
            }
            output_file = args.output / "prediction.json"
            output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"Counted {result.count} colonies; prediction: {output_file}")
            return 0
        if args.command == "evaluate":
            from colony_counting.evaluation import (
                MatchCriterion,
                counting_metrics,
                detection_map_metrics,
                detection_metrics,
                paired_bootstrap_mae_difference,
                paired_wilcoxon_mae,
                read_predictions,
            )

            records = read_predictions(args.predictions)
            criterion = MatchCriterion(
                args.match_mode, args.match_threshold, args.require_category_match
            )
            results = {
                "prediction_file": str(args.predictions),
                "counting": counting_metrics(
                    records,
                    absolute_tolerance=args.absolute_tolerance,
                    relative_tolerance=args.relative_tolerance,
                ),
                "detection": detection_metrics(records, criterion),
            }
            if all(
                box.confidence is not None for record in records for box in record.predicted_boxes
            ):
                results["detection_map"] = detection_map_metrics(
                    records, class_agnostic=not args.map_class_aware
                )
            else:
                results["detection_map"] = {
                    "status": "NOT_APPLICABLE",
                    "detail": "Saved prediction boxes do not contain confidence scores.",
                }
            if args.compare is not None:
                compared = read_predictions(args.compare)
                results["paired_comparison"] = {
                    "first": str(args.predictions),
                    "second": str(args.compare),
                    "mae_bootstrap": paired_bootstrap_mae_difference(
                        records, compared, resamples=args.bootstrap_resamples, seed=args.seed
                    ),
                    "wilcoxon_absolute_error": paired_wilcoxon_mae(records, compared),
                }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
            temp_output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
            temp_output.replace(args.output)
            print(f"Evaluation written to {args.output}")
            return 0
        if args.command == "analyze":
            from colony_counting.analysis import write_error_analysis
            from colony_counting.evaluation import read_predictions

            records = read_predictions(args.predictions)
            metadata = _load_split_metadata(args.manifest, records, args.split)
            report = write_error_analysis(
                records,
                metadata,
                args.output,
                images_root=args.images_root,
                gallery_size=args.gallery_size,
            )
            report.update(
                {
                    "prediction_file": str(args.predictions.resolve()),
                    "prediction_sha256": _file_sha256(args.predictions),
                    "manifest_file": str(args.manifest.resolve()),
                    "manifest_sha256": _file_sha256(args.manifest),
                    "split": args.split,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json_atomic(report, args.output / "analysis.json")
            print(
                f"Analyzed {report['n_images']} saved {args.split} predictions; "
                f"report: {args.output / 'analysis.json'}"
            )
            return 0
        if args.command == "ablations":
            from colony_counting.evaluation import (
                counting_metrics,
                paired_bootstrap_mae_difference,
                paired_group_bootstrap_mae_difference,
                paired_wilcoxon_mae,
                read_predictions,
            )

            baseline = read_predictions(args.baseline)
            metadata = _load_split_metadata(args.manifest, baseline, args.split)
            groups = {image_id: str(row["group_id"]) for image_id, row in metadata.items()}
            variants = dict(_parse_named_paths(args.variant))
            if not variants:
                raise ValueError("At least one --variant NAME=PATH is required.")
            comparisons = {}
            for name, prediction_path in sorted(variants.items()):
                variant = read_predictions(prediction_path)
                comparison: dict[str, object] = {
                    "prediction_file": str(prediction_path),
                    "prediction_sha256": _file_sha256(prediction_path),
                    "counting_metrics": counting_metrics(variant),
                    "mae_bootstrap_image": paired_bootstrap_mae_difference(
                        baseline,
                        variant,
                        resamples=args.bootstrap_resamples,
                        seed=args.seed,
                    ),
                    "paired_wilcoxon_absolute_error": paired_wilcoxon_mae(baseline, variant),
                }
                if len(set(groups.values())) > 1:
                    comparison["mae_bootstrap_group"] = paired_group_bootstrap_mae_difference(
                        baseline,
                        variant,
                        groups,
                        resamples=args.bootstrap_resamples,
                        seed=args.seed,
                    )
                else:
                    comparison["mae_bootstrap_group"] = {
                        "status": "not computed: only one group is represented"
                    }
                comparisons[name] = comparison
            payload = {
                "run_id": f"ablation-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "split": args.split,
                "baseline_prediction_file": str(args.baseline),
                "baseline_prediction_sha256": _file_sha256(args.baseline),
                "baseline_counting_metrics": counting_metrics(baseline),
                "baseline_n_images": len(baseline),
                "manifest_file": str(args.manifest.resolve()),
                "manifest_sha256": _file_sha256(args.manifest),
                "bootstrap_resamples": args.bootstrap_resamples,
                "seed": args.seed,
                "variant_comparisons": comparisons,
                "interpretation": (
                    "These are paired comparisons of precomputed outputs. This command does not "
                    "select thresholds, parameters, or models."
                ),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp = args.output.with_suffix(args.output.suffix + ".tmp")
            temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(args.output)
            print(f"Saved paired ablation comparisons to {args.output}")
            return 0
        if args.command == "tune-classical":
            from colony_counting.tuning import tune_classical

            metadata = tune_classical(
                manifest_path=args.manifest,
                images_root=args.images_root,
                config_path=args.config,
                output_dir=args.output,
                trials=args.trials,
                seed=args.seed,
                required_split="val",
            )
            print(
                f"Validation Optuna tuning completed {metadata['completed_trials']} trials; "
                f"best MAE: {metadata['best_validation_mae']:.6f}; "
                f"metadata: {args.output / 'experiment.json'}"
            )
            return 0
        if args.command == "tune-thresholds":
            from colony_counting.evaluation import (
                MatchCriterion,
                read_predictions,
                select_detection_thresholds,
            )

            records = read_predictions(args.predictions)
            _load_split_metadata(args.manifest, records, "val")
            result = select_detection_thresholds(
                records,
                args.thresholds,
                MatchCriterion(args.match_mode, args.match_threshold, args.require_category_match),
            )
            result.update(
                {
                    "prediction_file": str(args.predictions),
                    "manifest_file": str(args.manifest),
                    "manifest_sha256": _file_sha256(args.manifest),
                    "prediction_sha256": _file_sha256(args.predictions),
                    "selection_split": "val",
                    "run_id": f"thresholds-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "n_images": len(records),
                    "note": "Predictions must be generated at a lower confidence floor than all candidates.",
                }
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temp = args.output.with_suffix(args.output.suffix + ".tmp")
            temp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(args.output)
            print(
                f"Validation thresholds saved: count-MAE={result['threshold_count_mae']:.3f}, "
                f"detection-F1={result['threshold_detection_f1']:.3f} to {args.output}"
            )
            return 0
        if args.command in {"predict-classical", "predict-yolo"}:
            from colony_counting.manifest_inference import (
                predict_classical_manifest,
                predict_yolo_manifest,
            )

            if args.command == "predict-classical":
                run = predict_classical_manifest(
                    manifest_path=args.manifest,
                    images_root=args.images_root,
                    config_path=args.config,
                    split=args.split,
                    output_path=args.output,
                )
            else:
                run = predict_yolo_manifest(
                    manifest_path=args.manifest,
                    images_root=args.images_root,
                    config_path=args.config,
                    checkpoint=args.checkpoint,
                    split=args.split,
                    output_path=args.output,
                    confidence_floor=args.confidence_floor,
                    image_size=args.image_size,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    nms_iou=args.nms_iou,
                    device=args.device,
                    class_agnostic_nms=not args.class_aware_nms,
                    tiled=not args.whole_image,
                )
            print(
                f"Generated {run['n_images']} {run['method']} predictions on {run['split']} split; "
                f"file: {run['prediction_file']}"
            )
            return 0
        if args.command == "train-yolo":
            from colony_counting.detection.yolo import train_yolo

            metadata_path = train_yolo(
                dataset_yaml=args.data,
                weights=args.weights,
                output_dir=args.output,
                epochs=args.epochs,
                image_size=args.image_size,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=args.seed,
                workers=args.workers,
                device=args.device,
                run_name=args.name,
            )
            print(f"YOLO experiment metadata: {metadata_path}")
            return 0
        if args.command == "infer-yolo":
            import cv2

            from colony_counting.detection.yolo import infer_yolo, save_inference_result

            image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Cannot decode image: {args.image}", file=sys.stderr)
                return 1
            started = time.perf_counter()
            detections = infer_yolo(
                image=image,
                checkpoint=args.weights,
                tile_size=args.tile_size,
                overlap=args.overlap,
                confidence=args.confidence,
                nms_iou=args.nms_iou,
                image_size=args.image_size,
                device=args.device,
                class_agnostic_nms=not args.class_aware_nms,
                plate_circle=tuple(args.plate_circle) if args.plate_circle else None,
                plate_roi_scale=args.plate_roi_scale,
                tiled=not args.whole_image,
            )
            elapsed = time.perf_counter() - started
            prediction_path, overlay_path = save_inference_result(
                image,
                detections,
                checkpoint=args.weights,
                output_dir=args.output,
                inference_seconds=elapsed,
                tiled=not args.whole_image,
                confidence=args.confidence,
                tile_size=None if args.whole_image else args.tile_size,
                overlap=None if args.whole_image else args.overlap,
                nms_iou=args.nms_iou,
                device=args.device,
                class_agnostic_nms=not args.class_aware_nms,
                plate_circle=tuple(args.plate_circle) if args.plate_circle else None,
            )
            print(
                f"YOLO predicted {len(detections)} objects in {elapsed:.4f}s; "
                f"prediction: {prediction_path}; overlay: {overlay_path}"
            )
            return 0
        if args.command == "app":
            from colony_counting.app import launch_app

            launch_app(
                config_path=args.config,
                checkpoint=args.checkpoint,
                server_name=args.server_name,
                server_port=args.server_port,
            )
            return 0
        if args.command == "setup":
            print(
                "Create an isolated Python 3.11 or 3.12 environment, then install: pip install -e .[dev]"
            )
            return 0
        if args.command == "smoke":
            from colony_counting.classical import ClassicalConfig, load_classical_config
            from colony_counting.evaluation import MatchCriterion

            ClassicalConfig()
            MatchCriterion()
            load_classical_config(Path("configs/classical.yaml"))
            print("Core imports and classical configuration parsing succeeded.")
            return 0
        if args.command == "test":
            import pytest

            return int(pytest.main(["-q", "tests"]))
        if args.command == "lint":
            try:
                from ruff import __main__ as ruff_main  # noqa: F401
            except ImportError:
                print("ruff is not installed; install the dev extra.", file=sys.stderr)
                return 2
            import subprocess

            return subprocess.call(
                [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]
            )
        if args.command == "verify":
            return _run_verification(args.output)
        print(
            f"The '{args.command}' task is not implemented yet; see harness/PLAN.md.",
            file=sys.stderr,
        )
        return 2
    except (DatasetFormatError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Task failed: {exc}", file=sys.stderr)
        return 1


def _run_verification(output_file: Path) -> int:
    """Run local software gates and write a report that exposes missing experiment assets."""

    checks: dict[str, dict[str, str]] = {}
    modules = (
        "numpy",
        "cv2",
        "PIL",
        "yaml",
        "scipy",
        "matplotlib",
        "pytest",
        "ruff",
        "colony_counting.analysis",
        "colony_counting.app",
        "colony_counting.classical",
        "colony_counting.detection.tiling",
        "colony_counting.detection.yolo",
        "colony_counting.detection.yolo_data",
        "colony_counting.evaluation",
        "colony_counting.manifest_inference",
        "colony_counting.tuning",
    )
    for module_name in modules:
        try:
            __import__(module_name)
            checks[f"import:{module_name}"] = {"status": "PASS", "detail": "import succeeded"}
        except (ImportError, OSError, RuntimeError) as exc:
            checks[f"import:{module_name}"] = {"status": "FAIL", "detail": str(exc)}

    try:
        from colony_counting.classical import load_classical_config

        load_classical_config(Path("configs/classical.yaml"))
        checks["config:classical"] = {"status": "PASS", "detail": "YAML parsed and validated"}
    except (OSError, TypeError, ValueError) as exc:
        checks["config:classical"] = {"status": "FAIL", "detail": str(exc)}

    for name, command in (
        ("unit_tests", [sys.executable, "-m", "pytest", "-q"]),
        ("lint", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"]),
        ("smoke", [sys.executable, "-m", "colony_counting", "smoke"]),
    ):
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        checks[name] = {
            "status": "PASS" if run.returncode == 0 else "FAIL",
            "exit_code": str(run.returncode),
            "detail": (run.stdout + run.stderr).strip()[-4000:],
        }

    manifest_path = Path("data/processed/manifest.jsonl")
    if manifest_path.is_file():
        try:
            _validate_manifest_structure(manifest_path)
            checks["dataset_manifest"] = {
                "status": "PASS",
                "detail": f"JSONL structure and group split integrity checked: {manifest_path}",
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            checks["dataset_manifest"] = {"status": "FAIL", "detail": str(exc)}
    else:
        checks["dataset_manifest"] = {
            "status": "BLOCKED",
            "detail": "No local data/processed/manifest.jsonl exists",
        }
    model_path = Path("artifacts/yolo/best.pt")
    checks["yolo_checkpoint"] = {
        "status": "PASS" if model_path.is_file() else "BLOCKED",
        "detail": str(model_path) if model_path.is_file() else "No trained YOLO checkpoint exists",
    }
    checks["gradio"] = {
        "status": "PASS" if importlib.util.find_spec("gradio") else "BLOCKED",
        "detail": "Installed" if importlib.util.find_spec("gradio") else "Gradio is not installed",
    }
    checks["optuna"] = {
        "status": "PASS" if importlib.util.find_spec("optuna") else "BLOCKED",
        "detail": "Installed" if importlib.util.find_spec("optuna") else "Optuna is not installed",
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        checks["cuda"] = {
            "status": "PASS" if cuda_available else "BLOCKED",
            "detail": (
                torch.cuda.get_device_name(0)
                if cuda_available
                else f"CUDA unavailable in torch {torch.__version__}"
            ),
        }
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        checks["cuda"] = {"status": "BLOCKED", "detail": f"Cannot inspect torch CUDA: {exc}"}

    required_code_gates = (
        "import:numpy",
        "import:cv2",
        "import:PIL",
        "import:yaml",
        "import:colony_counting.analysis",
        "import:colony_counting.app",
        "import:colony_counting.classical",
        "import:colony_counting.evaluation",
        "import:colony_counting.manifest_inference",
        "import:colony_counting.tuning",
        "config:classical",
        "unit_tests",
        "lint",
        "smoke",
    )
    code_gates_pass = all(checks[name]["status"] == "PASS" for name in required_code_gates)
    report = {
        "verification_level": "UNIT_VERIFIED" if code_gates_pass else "FAILED",
        "project_verified": False,
        "checks": checks,
        "note": "Code gates do not establish real-data experiments or full project acceptance.",
    }
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    temp_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temp_file.replace(output_file)
    print(
        f"Verification report: {output_file} ({report['verification_level']}); project_verified=false"
    )
    return 0 if code_gates_pass else 1


def _validate_manifest_structure(manifest_file: Path) -> None:
    """Check manifest JSONL row IDs, groups, and split assignment without assuming data paths."""

    seen_ids: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    hash_splits: dict[str, set[str]] = {}
    for line_number, line in enumerate(manifest_file.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"Manifest row {line_number} must be a JSON object")
        if row.get("image_id") is None or row.get("group_id") is None:
            raise ValueError(f"Manifest row {line_number} is missing image_id or group_id")
        image_id = str(row.get("image_id", ""))
        group_id = str(row.get("split_group_id") or row.get("group_id", ""))
        split = row.get("split")
        if not image_id or not group_id:
            raise ValueError(f"Manifest row {line_number} is missing image_id or group_id")
        if image_id in seen_ids:
            raise ValueError(f"Duplicate manifest image_id {image_id!r}")
        seen_ids.add(image_id)
        if split is not None and split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {split!r} on manifest row {line_number}")
        group_splits.setdefault(group_id, set()).add(split)
        if row.get("sha256"):
            hash_splits.setdefault(str(row["sha256"]), set()).add(split)
    leaking = sorted(group_id for group_id, values in group_splits.items() if len(values) > 1)
    leaking.extend(
        f"duplicate-sha256:{content_hash}"
        for content_hash, values in hash_splits.items()
        if len(values) > 1
    )
    if leaking:
        raise ValueError(f"Groups assigned across multiple partitions: {leaking[:10]}")


def _load_split_metadata(
    manifest_file: Path,
    records: list[object],
    required_split: str,
) -> dict[str, dict[str, object]]:
    """Load metadata and ensure every prediction belongs to the requested frozen split."""

    rows: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(manifest_file.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("image_id") is None:
            raise ValueError(f"Invalid manifest row {line_number}: expected image_id object.")
        image_id = str(row["image_id"])
        if image_id in rows:
            raise ValueError(f"Duplicate manifest image_id {image_id!r}.")
        rows[image_id] = row
    prediction_ids = {record.image_id for record in records}
    missing = sorted(prediction_ids - rows.keys())
    if missing:
        raise ValueError(f"Predictions reference IDs absent from manifest: {missing[:10]}.")
    wrong_split = sorted(
        image_id for image_id in prediction_ids if rows[image_id].get("split") != required_split
    )
    if wrong_split:
        raise ValueError(
            f"Predictions must all belong to requested split {required_split!r}; "
            f"mismatches={wrong_split[:10]}."
        )
    return {image_id: rows[image_id] for image_id in prediction_ids}


def _parse_named_paths(items: list[str]) -> list[tuple[str, Path]]:
    """Parse repeated CLI name=path entries, rejecting duplicates and empty values."""

    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in items:
        name, separator, raw_path = item.partition("=")
        name, raw_path = name.strip(), raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(f"Expected --variant NAME=PATH, got {item!r}.")
        if name in seen:
            raise ValueError(f"Duplicate ablation variant name {name!r}.")
        seen.add(name)
        parsed.append((name, Path(raw_path)))
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(payload: dict[str, object], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    raise SystemExit(main())
