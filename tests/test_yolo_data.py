"""YOLO label conversion tests with generated files, not AGAR observations."""

import pytest
import yaml
from PIL import Image

from colony_counting.data.coco import Box, ImageRecord
from colony_counting.detection.yolo_data import export_yolo_dataset


def _records():
    return [
        ImageRecord(
            "countable",
            "countable.png",
            200,
            100,
            (Box(20, 10, 40, 20, 3, "species-a"), Box(100, 40, 20, 20, 8, "species-b")),
            "plate-a",
            "ring",
            "camera-1",
            "countable",
            "train",
        ),
        ImageRecord("empty", "empty.png", 200, 100, (), "plate-b", None, None, "empty", "val"),
    ]


def test_class_agnostic_yolo_conversion_and_empty_label(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (200, 100)).save(image_root / "countable.png")
    Image.new("RGB", (200, 100)).save(image_root / "empty.png")
    output = tmp_path / "export"
    config_file = export_yolo_dataset(_records(), image_root, output)
    label_lines = (output / "labels" / "train" / "countable.txt").read_text().splitlines()
    assert label_lines == [
        "0 0.20000000 0.20000000 0.20000000 0.20000000",
        "0 0.55000000 0.50000000 0.10000000 0.20000000",
    ]
    assert (output / "labels" / "val" / "empty.txt").read_text() == ""
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert config["train"] == "images/train"
    assert config["names"] == {0: "colony"}
    assert (output / "images" / "train" / "countable.png").is_file()


def test_species_aware_class_ids_are_stable(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (200, 100)).save(image_root / "countable.png")
    Image.new("RGB", (200, 100)).save(image_root / "empty.png")
    config_file = export_yolo_dataset(
        _records(), image_root, tmp_path / "species", class_agnostic=False
    )
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert config["names"] == {0: "species-a", 1: "species-b"}
    assert (
        (tmp_path / "species" / "labels" / "train" / "countable.txt").read_text().startswith("0 ")
    )


def test_requires_frozen_splits_and_existing_images(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    records = _records()
    with pytest.raises(ValueError, match="frozen split"):
        export_yolo_dataset(
            [ImageRecord("i", "x.png", 2, 2, (Box(0, 0, 1, 1, 0, "c"),), "g", None, None, None)],
            image_root,
            tmp_path / "out",
        )
    with pytest.raises(FileNotFoundError, match="missing"):
        export_yolo_dataset(records, image_root, tmp_path / "out")
