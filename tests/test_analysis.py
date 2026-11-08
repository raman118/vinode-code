import json

import cv2
import numpy as np
import pytest

from colony_counting.analysis import stratified_count_metrics, write_error_analysis
from colony_counting.cli import main
from colony_counting.evaluation import (
    EvalBox,
    PredictionRecord,
    paired_group_bootstrap_mae_difference,
    write_predictions,
)


def _records():
    return [
        PredictionRecord(
            "a",
            0,
            1,
            (),
            (EvalBox(4, 4, 8, 8),),
        ),
        PredictionRecord(
            "b",
            2,
            2,
            (EvalBox(5, 5, 8, 8), EvalBox(22, 20, 7, 8)),
            (EvalBox(5, 5, 8, 8), EvalBox(22, 20, 7, 8)),
        ),
        PredictionRecord("c", 25, 20),
    ]


def test_stratification_reports_only_populated_bins_and_observed_metadata():
    records = _records()
    metadata = {
        "a": {"lighting": "L1", "status": "empty", "camera": None},
        "b": {"lighting": "L2", "status": "countable", "camera": "cam-a"},
        "c": {"lighting": "L2", "status": "countable", "camera": "cam-a"},
    }
    result = stratified_count_metrics(records, metadata)

    assert result["count_bins"]["empty"]["n_images"] == 1
    assert result["count_bins"]["1-20"]["n_images"] == 1
    assert result["count_bins"]["21-50"]["n_images"] == 1
    assert result["metadata_strata"]["lighting"]["L2"]["n_images"] == 2
    assert result["metadata_strata"]["camera"]["cam-a"]["n_images"] == 2


def test_stratification_requires_manifest_metadata_for_every_prediction():
    with pytest.raises(ValueError, match="Missing metadata"):
        stratified_count_metrics(_records(), {"a": {}})


def test_group_bootstrap_is_reproducible_and_uses_group_resampling():
    baseline = _records()
    variant = [
        PredictionRecord(
            record.image_id, record.gt_count, record.gt_count, record.gt_boxes, record.gt_boxes
        )
        for record in baseline
    ]
    groups = {"a": "plate-1", "b": "plate-1", "c": "plate-2"}
    first = paired_group_bootstrap_mae_difference(baseline, variant, groups, resamples=250, seed=19)
    second = paired_group_bootstrap_mae_difference(
        baseline, variant, groups, resamples=250, seed=19
    )
    assert first == second
    assert first["n_groups"] == 2
    assert first["resampling_unit"] == "group"


def test_error_analysis_writes_plots_and_overlay_gallery(tmp_path):
    records = _records()
    image_root = tmp_path / "images"
    image_root.mkdir()
    metadata = {}
    for record in records:
        path = f"{record.image_id}.png"
        image = np.full((48, 48, 3), 220, dtype=np.uint8)
        assert cv2.imwrite(str(image_root / path), image)
        metadata[record.image_id] = {
            "image_path": path,
            "lighting": "fixture-light",
            "status": "synthetic-test",
        }

    output = tmp_path / "analysis"
    report = write_error_analysis(records, metadata, output, images_root=image_root)

    assert report["status"] == "computed_from_saved_predictions"
    assert report["human_failure_labels"].startswith("not collected")
    assert report["gallery_status"] == "3/3 automatically ranked images rendered"
    assert (output / "analysis.json").is_file()
    assert (output / "count_error_figures.png").is_file()
    assert len(list((output / "worst_cases").glob("*.jpg"))) == 3
    stored = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert stored["overall_counting"]["n_images"] == 3


def test_analysis_and_ablation_cli_reject_cross_split_and_write_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline = _records()
    perfect = [
        PredictionRecord(
            record.image_id, record.gt_count, record.gt_count, record.gt_boxes, record.gt_boxes
        )
        for record in baseline
    ]
    predictions_dir = tmp_path / "predictions"
    write_predictions(baseline, predictions_dir / "baseline.jsonl")
    write_predictions(perfect, predictions_dir / "perfect.jsonl")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"image_id": image_id, "group_id": group, "split": "test", "lighting": lighting}
        for image_id, group, lighting in (
            ("a", "plate-1", "L1"),
            ("b", "plate-1", "L1"),
            ("c", "plate-2", "L2"),
        )
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert (
        main(
            [
                "analyze",
                "--predictions",
                str(predictions_dir / "baseline.jsonl"),
                "--manifest",
                str(manifest),
                "--split",
                "test",
                "--output",
                "analysis-output",
            ]
        )
        == 0
    )
    assert (tmp_path / "analysis-output" / "analysis.json").is_file()
    assert (
        main(
            [
                "ablations",
                "--baseline",
                str(predictions_dir / "baseline.jsonl"),
                "--variant",
                f"perfect={predictions_dir / 'perfect.jsonl'}",
                "--manifest",
                str(manifest),
                "--split",
                "test",
                "--output",
                "ablations.json",
                "--bootstrap-resamples",
                "50",
            ]
        )
        == 0
    )
    ablation = json.loads((tmp_path / "ablations.json").read_text(encoding="utf-8"))
    comparison = ablation["variant_comparisons"]["perfect"]
    assert comparison["mae_bootstrap_group"]["n_groups"] == 2
    assert comparison["mae_bootstrap_image"]["n_images"] == 3

    assert (
        main(
            [
                "analyze",
                "--predictions",
                str(predictions_dir / "baseline.jsonl"),
                "--manifest",
                str(manifest),
                "--split",
                "val",
                "--output",
                "wrong-split",
            ]
        )
        == 1
    )


def test_tune_thresholds_cli_enforces_validation_and_writes_curve(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record = PredictionRecord(
        "val-image",
        1,
        2,
        (EvalBox(2, 2, 10, 10),),
        (
            EvalBox(2, 2, 10, 10, confidence=0.8),
            EvalBox(24, 24, 6, 6, confidence=0.6),
        ),
    )
    predictions = tmp_path / "validation.jsonl"
    write_predictions([record], predictions)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"image_id": "val-image", "group_id": "g1", "split": "val"}) + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "tune-thresholds",
                "--predictions",
                str(predictions),
                "--manifest",
                str(manifest),
                "--thresholds",
                "0.5",
                "0.7",
                "0.9",
                "--output",
                "thresholds.json",
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "thresholds.json").read_text(encoding="utf-8"))
    assert payload["selection_split"] == "val"
    assert payload["threshold_count_mae"] == 0.7
    assert payload["threshold_detection_f1"] == 0.7
    assert len(payload["threshold_curve"]) == 3
