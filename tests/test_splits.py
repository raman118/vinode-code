"""Tests for grouped split logic; fixtures are synthetic, not AGAR results."""

from dataclasses import replace

import pytest

from colony_counting.data.coco import ImageRecord
from colony_counting.data.splits import split_by_group, validate_group_splits


def _records():
    return [
        ImageRecord(i, f"{i}.png", 20, 20, (), f"plate-{i // 2}", None, None, None)
        for i in range(40)
    ]


def test_split_is_seeded_and_group_safe():
    records = _records()
    first = split_by_group(records, seed=13)
    second = split_by_group(records, seed=13)
    assert [record.split for record in first] == [record.split for record in second]
    validate_group_splits(first)
    for plate in {record.group_id for record in first}:
        assert len({record.split for record in first if record.group_id == plate}) == 1
    assert {record.split for record in first} == {"train", "val", "test"}


def test_split_rejects_duplicate_image_ids():
    records = _records()
    records[1] = replace(records[1], image_id=records[0].image_id)
    with pytest.raises(ValueError, match="unique"):
        split_by_group(records)


def test_validator_detects_group_leakage():
    records = _records()[:2]
    records = [replace(records[0], split="train"), replace(records[1], split="test")]
    with pytest.raises(ValueError, match="multiple partitions"):
        validate_group_splits(records)


def test_split_co_locates_byte_identical_images_with_different_plate_ids():
    records = [
        ImageRecord(1, "a.png", 20, 20, (), "plate-a", None, None, None, sha256="same"),
        ImageRecord(2, "b.png", 20, 20, (), "plate-b", None, None, None, sha256="same"),
        *[
            ImageRecord(i, f"{i}.png", 20, 20, (), f"plate-{i}", None, None, None)
            for i in range(3, 13)
        ],
    ]
    assigned = split_by_group(records, seed=91)
    assert assigned[0].split == assigned[1].split
    assert assigned[0].split_group_id == assigned[1].split_group_id


def test_empty_input_is_supported():
    assert split_by_group([]) == []
