"""Parse and validate a COCO-style detection dataset into stable image records.

This is a generic normalized input adapter. It does not assume the raw AGAR release uses COCO.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class Box:
    """An object box in absolute XYWH pixel coordinates."""

    x: float
    y: float
    width: float
    height: float
    category_id: int
    category_name: str


@dataclass(frozen=True)
class ImageRecord:
    """Normalized metadata for one image and all associated annotations."""

    image_id: int | str
    image_path: str
    width: int
    height: int
    boxes: tuple[Box, ...]
    group_id: str
    lighting: str | None
    camera: str | None
    status: str | None
    split: str | None = None
    sha256: str | None = None
    split_group_id: str | None = None

    @property
    def count(self) -> int:
        """Return the number of object boxes attached to this image."""

        return len(self.boxes)


class DatasetFormatError(ValueError):
    """Raised when source annotations or image files violate the normalized contract."""


def load_coco_records(annotation_file: Path, image_root: Path) -> list[ImageRecord]:
    """Load a COCO JSON file and validate IDs, paths, dimensions, and every XYWH box.

    Optional image fields ``group_id``, ``lighting``, ``camera``, and ``status`` are retained.
    Missing group IDs conservatively fall back to the image ID, so no unproven grouping is inferred.
    """

    annotation_file = Path(annotation_file)
    image_root = Path(image_root)
    try:
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"Cannot read annotation JSON {annotation_file}: {exc}") from exc

    if not isinstance(payload, dict):
        raise DatasetFormatError("Annotation root must be a JSON object.")
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise DatasetFormatError("Expected top-level 'images' and 'annotations' arrays.")

    category_names: dict[int, str] = {}
    for category in categories:
        try:
            category_id = int(category["id"])
            category_name = str(category["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetFormatError(f"Invalid category entry: {category!r}") from exc
        if category_id in category_names:
            raise DatasetFormatError(f"Duplicate category id {category_id}.")
        category_names[category_id] = category_name

    by_image: dict[int | str, list[Box]] = {}
    for annotation in annotations:
        try:
            image_id = annotation["image_id"]
            category_id = int(annotation["category_id"])
            x, y, width, height = (float(value) for value in annotation["bbox"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetFormatError(f"Invalid annotation entry: {annotation!r}") from exc
        if category_names and category_id not in category_names:
            raise DatasetFormatError(f"Annotation references unknown category id {category_id}.")
        if not all(map(_finite, (x, y, width, height))) or width <= 0 or height <= 0:
            raise DatasetFormatError(
                f"Invalid non-positive or non-finite box {annotation.get('bbox')}."
            )
        by_image.setdefault(image_id, []).append(
            Box(x, y, width, height, category_id, category_names.get(category_id, str(category_id)))
        )

    records: list[ImageRecord] = []
    seen_ids: set[int | str] = set()
    file_hashes: dict[Path, str] = {}
    for image in images:
        try:
            image_id = image["id"]
            rel_path = Path(str(image["file_name"]))
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetFormatError(f"Invalid image entry: {image!r}") from exc
        if image_id in seen_ids:
            raise DatasetFormatError(f"Duplicate image id {image_id!r}.")
        seen_ids.add(image_id)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise DatasetFormatError(
                f"Image path must be relative and stay in image root: {rel_path}"
            )
        image_path = image_root / rel_path
        if not image_path.is_file():
            raise DatasetFormatError(
                f"Image file referenced by annotations is missing: {image_path}"
            )
        if width <= 0 or height <= 0:
            raise DatasetFormatError(f"Invalid image dimensions for {rel_path}: {width}x{height}.")
        try:
            with Image.open(image_path) as opened_image:
                actual_width, actual_height = opened_image.size
        except OSError as exc:
            raise DatasetFormatError(f"Cannot decode image {image_path}: {exc}") from exc
        if (actual_width, actual_height) != (width, height):
            raise DatasetFormatError(
                f"Image dimensions disagree for {rel_path}: annotation says {width}x{height}, "
                f"file is {actual_width}x{actual_height}."
            )
        boxes = by_image.pop(image_id, [])
        for box in boxes:
            if box.x < 0 or box.y < 0 or box.x + box.width > width or box.y + box.height > height:
                raise DatasetFormatError(
                    f"Box {box} falls outside annotated dimensions for {rel_path}."
                )
        if image_path not in file_hashes:
            file_hashes[image_path] = _sha256_file(image_path)
        records.append(
            ImageRecord(
                image_id=image_id,
                image_path=rel_path.as_posix(),
                width=width,
                height=height,
                boxes=tuple(boxes),
                group_id=str(image.get("group_id", image_id)),
                lighting=_optional_text(image.get("lighting")),
                camera=_optional_text(image.get("camera")),
                status=_optional_text(image.get("status")),
                sha256=file_hashes[image_path],
            )
        )
    if by_image:
        orphan_ids = sorted(map(str, by_image))
        raise DatasetFormatError(
            f"Annotations reference image IDs absent from images: {orphan_ids[:10]}"
        )
    return records


def write_manifest(records: list[ImageRecord], output_file: Path) -> None:
    """Write deterministic JSONL records, including explicit box lists and counts."""

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temp_file.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            row: dict[str, Any] = asdict(record)
            row["boxes"] = [asdict(box) for box in record.boxes]
            row["count"] = record.count
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp_file.replace(output_file)


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _optional_text(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
