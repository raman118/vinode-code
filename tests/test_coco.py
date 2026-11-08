"""Tests for generic COCO ingestion using tiny synthetic fixtures, not AGAR data."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from colony_counting.data.coco import DatasetFormatError, load_coco_records, write_manifest


def _fixture(tmp_path, bbox=(1, 1, 3, 4), dimensions=(10, 8)):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", dimensions).save(image_root / "plate.png")
    payload = {
        "images": [
            {
                "id": 7,
                "file_name": "plate.png",
                "width": dimensions[0],
                "height": dimensions[1],
                "group_id": "plate-a",
                "lighting": "top",
                "camera": "cam-x",
            },
            {"id": 8, "file_name": "plate.png", "width": dimensions[0], "height": dimensions[1]},
        ],
        "annotations": [{"id": 1, "image_id": 7, "category_id": 2, "bbox": bbox}],
        "categories": [{"id": 2, "name": "colony"}],
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    return annotation_path, image_root


def test_loads_records_and_preserves_empty_images(tmp_path):
    annotation_path, image_root = _fixture(tmp_path)
    records = load_coco_records(annotation_path, image_root)
    assert [record.count for record in records] == [1, 0]
    assert records[0].group_id == "plate-a"
    assert records[0].lighting == "top"
    assert records[0].boxes[0].category_name == "colony"
    assert records[0].sha256 == records[1].sha256


@pytest.mark.parametrize("bbox", [(8, 1, 3, 2), (1, 1, 0, 2), (1, 1, float("nan"), 2)])
def test_rejects_invalid_boxes(tmp_path, bbox):
    annotation_path, image_root = _fixture(tmp_path, bbox=bbox)
    with pytest.raises(DatasetFormatError):
        load_coco_records(annotation_path, image_root)


def test_rejects_dimension_mismatch(tmp_path):
    annotation_path, image_root = _fixture(tmp_path, dimensions=(10, 9))
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["images"][0]["height"] = 8
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="dimensions disagree"):
        load_coco_records(annotation_path, image_root)


def test_rejects_path_traversal(tmp_path):
    annotation_path, image_root = _fixture(tmp_path)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = "../plate.png"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="relative"):
        load_coco_records(annotation_path, image_root)


def test_manifest_is_jsonl_and_counts_match(tmp_path):
    annotation_path, image_root = _fixture(tmp_path)
    records = load_coco_records(annotation_path, image_root)
    output = tmp_path / "manifest.jsonl"
    write_manifest(records, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["count"] for row in rows] == [1, 0]
    assert rows[1]["boxes"] == []
