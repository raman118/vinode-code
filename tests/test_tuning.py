import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from colony_counting.cli import main


def test_tune_classical_only_reads_validation_and_persists_trial_provenance(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    monkeypatch.chdir(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    validation_image = np.full((96, 96, 3), 225, dtype=np.uint8)
    cv2.circle(validation_image, (35, 40), 7, (20, 20, 20), -1)
    cv2.circle(validation_image, (62, 59), 7, (20, 20, 20), -1)
    assert cv2.imwrite(str(images / "validation.png"), validation_image)
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "image_id": "val-1",
            "image_path": "validation.png",
            "count": 2,
            "split": "val",
        },
        {
            "image_id": "test-1",
            "image_path": "intentionally-missing-test-file.png",
            "count": 999,
            "split": "test",
        },
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    exit_code = main(
        [
            "tune-classical",
            "--manifest",
            str(manifest),
            "--images-root",
            str(images),
            "--config",
            str(Path(__file__).parents[1] / "configs" / "classical.yaml"),
            "--output",
            "tuning-output",
            "--trials",
            "1",
            "--seed",
            "27",
        ]
    )

    assert exit_code == 0
    output = tmp_path / "tuning-output"
    metadata = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert metadata["required_split"] == "val"
    assert metadata["validation_images"] == 1
    assert metadata["completed_trials"] == 1
    assert metadata["seed"] == 27
    assert Path(metadata["best_config"]).is_file()
    assert Path(metadata["objective_history_csv"]).is_file()
