import json
from pathlib import Path

import cv2
import numpy as np

from colony_counting.cli import main
from colony_counting.detection.tiling import DetectionBox
from colony_counting.evaluation import read_predictions


def _manifest(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    image = np.full((80, 80, 3), 225, dtype=np.uint8)
    cv2.circle(image, (40, 40), 7, (20, 20, 20), -1)
    assert cv2.imwrite(str(images / "plate.png"), image)
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "image_id": "plate-1",
        "image_path": "plate.png",
        "width": 80,
        "height": 80,
        "count": 1,
        "boxes": [
            {
                "x": 33,
                "y": 33,
                "width": 14,
                "height": 14,
                "category_id": 0,
                "category_name": "colony",
            }
        ],
        "group_id": "group-1",
        "split": "val",
        "lighting": "light-1",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest, images


def test_classical_batch_cli_saves_split_predictions_and_run_provenance(tmp_path):
    manifest, images = _manifest(tmp_path)
    output = tmp_path / "classical.jsonl"
    config = Path(__file__).parents[1] / "configs" / "classical.yaml"

    assert (
        main(
            [
                "predict-classical",
                "--manifest",
                str(manifest),
                "--images-root",
                str(images),
                "--split",
                "val",
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    record = read_predictions(output)[0]
    assert record.image_id == "plate-1"
    assert record.gt_count == len(record.gt_boxes) == 1
    assert record.metadata["lighting"] == "light-1"
    run = json.loads(output.with_suffix(".run.json").read_text(encoding="utf-8"))
    assert run["method"] == "classical"
    assert run["split"] == "val"
    assert run["n_images"] == 1
    assert run["completed_at_utc"] > run["started_at_utc"]


def test_yolo_batch_cli_preserves_confidence_and_loads_predictor_once(tmp_path, monkeypatch):
    manifest, images = _manifest(tmp_path)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"synthetic test checkpoint marker")
    output = tmp_path / "yolo.jsonl"
    config = Path(__file__).parents[1] / "configs" / "classical.yaml"
    from colony_counting import manifest_inference

    loaded = []
    predictor = object()

    def fake_load(path, **kwargs):
        loaded.append((path, kwargs))
        return predictor

    def fake_infer(image, active_predictor, **kwargs):
        assert active_predictor is predictor
        return [DetectionBox(32, 32, 46, 46, 0.87, 0)]

    monkeypatch.setattr(manifest_inference, "load_yolo_predictor", fake_load)
    monkeypatch.setattr(manifest_inference, "infer_yolo_with_predictor", fake_infer)

    assert (
        main(
            [
                "predict-yolo",
                "--manifest",
                str(manifest),
                "--images-root",
                str(images),
                "--split",
                "val",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(loaded) == 1
    record = read_predictions(output)[0]
    assert record.predicted_count == len(record.predicted_boxes) == 1
    assert record.predicted_boxes[0].confidence == 0.87
    run = json.loads(output.with_suffix(".run.json").read_text(encoding="utf-8"))
    assert run["method"] == "yolo"
    assert run["checkpoint_sha256"]
