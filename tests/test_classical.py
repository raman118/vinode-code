"""Synthetic image tests for classical CV components, not AGAR performance evidence."""

import cv2
import numpy as np
import pytest

from colony_counting.classical import (
    ClassicalConfig,
    circular_roi_mask,
    clean_mask,
    count_colonies,
    detect_plate_circle,
    load_classical_config,
    save_diagnostics,
    select_channel,
)


def _synthetic_plate():
    image = np.full((120, 120, 3), 225, dtype=np.uint8)
    for center in ((35, 38), (82, 36), (59, 82)):
        cv2.circle(image, center, 6, (30, 30, 30), -1)
    return image


def test_count_colonies_on_three_separated_synthetic_disks(tmp_path):
    config = ClassicalConfig(
        illumination_correction=False,
        plate_detection=False,
        plate_roi_scale=1.0,
        peak_min_distance=4,
        peak_min_fraction=0.4,
        min_colony_area=10,
        max_colony_area=500,
        morphology_kernel=3,
    )
    result = count_colonies(_synthetic_plate(), config)
    assert result.count == 3
    assert set(result.stages) == {
        "roi",
        "channel",
        "illumination_corrected",
        "contrast_normalized",
        "threshold",
        "morphology",
        "distance_transform",
        "markers",
        "watershed",
        "overlay",
    }
    assert all(0 <= item.x < 120 and 0 <= item.y < 120 for item in result.detections)
    assert all(item.box[2] > 0 and item.box[3] > 0 for item in result.detections)
    saved = save_diagnostics(result, tmp_path / "diagnostics")
    assert len(saved) == 10


def test_components_are_independently_callable():
    image = _synthetic_plate()
    assert select_channel(image, "red").shape == image.shape[:2]
    assert detect_plate_circle(image, ClassicalConfig()) is None
    mask = circular_roi_mask((50, 60), None, 0.9)
    assert mask[25, 30] == 255
    assert mask[0, 0] == 0
    binary = np.zeros((20, 20), dtype=np.uint8)
    binary[5:10, 5:10] = 255
    assert clean_mask(binary, 3, 0, 0).sum() > 0


def test_yaml_config_load_and_invalid_parameters(tmp_path):
    config_file = tmp_path / "classical.yaml"
    config_file.write_text("channel: red\nplate_detection: false\n", encoding="utf-8")
    config = load_classical_config(config_file)
    assert config.channel == "red"
    assert not config.plate_detection
    with pytest.raises(ValueError, match="positive odd"):
        ClassicalConfig(morphology_kernel=2)


def test_rejects_invalid_image():
    with pytest.raises(ValueError, match="non-empty"):
        count_colonies(np.array([], dtype=np.uint8))
