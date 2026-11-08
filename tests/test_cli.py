"""CLI integration checks use generated toy images, not real plate data."""

import json

import cv2
import numpy as np
import pytest

from colony_counting.cli import _validate_manifest_structure, main
from colony_counting.evaluation import EvalBox, PredictionRecord, write_predictions


def test_classical_cli_writes_prediction_and_diagnostics(tmp_path):
    image = np.full((80, 80, 3), 225, dtype=np.uint8)
    cv2.circle(image, (40, 40), 7, (20, 20, 20), -1)
    input_path = tmp_path / "fixture.png"
    assert cv2.imwrite(str(input_path), image)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "illumination_correction: false\nplate_detection: false\nplate_roi_scale: 1.0\n"
        "peak_min_distance: 4\npeak_min_fraction: 0.4\nmin_colony_area: 10\nmax_colony_area: 500\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "out"
    exit_code = main(
        [
            "classical",
            "--image",
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads((output_path / "prediction.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["count"] == 1
    assert len(payload["diagnostic_files"]) == 10


def test_evaluate_cli_reads_saved_predictions(tmp_path):
    prediction_file = tmp_path / "predictions.jsonl"
    write_predictions(
        [
            PredictionRecord("plate-1", 1, 1, (EvalBox(1, 2, 5, 6),), (EvalBox(1, 2, 5, 6),)),
            PredictionRecord("plate-2", 0, 0),
        ],
        prediction_file,
    )
    output = tmp_path / "metrics.json"
    exit_code = main(
        [
            "evaluate",
            "--predictions",
            str(prediction_file),
            "--output",
            str(output),
            "--match-mode",
            "iou",
            "--match-threshold",
            "0.5",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["counting"]["mae"] == 0
    assert payload["detection"]["f1"] == 1


def test_manifest_structure_check_detects_group_leakage(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"image_id":"a","group_id":"plate-1","split":"train"}\n'
        '{"image_id":"b","group_id":"plate-1","split":"test"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple partitions"):
        _validate_manifest_structure(path)
