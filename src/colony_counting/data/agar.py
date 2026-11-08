"""Parse the AGAR per-image JSON format into normalised ImageRecord objects.

Each image in the AGAR Bright dataset lives alongside a sidecar JSON file
with this schema::

    {
        "background": "bright",
        "classes": ["B.subtilis"],
        "colonies_number": 4,          # -1 means uncountable
        "labels": [
            {"class": "B.subtilis", "height": 190, "id": 1,
             "width": 190, "x": 1099, "y": 1524},
            ...
        ],
        "sample_id": 1001
    }

Box coordinates ``x`` and ``y`` are the **top-left corner** in pixels.
``width`` and ``height`` are in pixels.  The status is inferred from the
directory name: ``countable``, ``empty``, or ``uncountable``.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from colony_counting.data.coco import Box, DatasetFormatError, ImageRecord, _sha256_file


_STATUS_MAP = {
    "countable": "countable",
    "empty": "empty",
    "uncountable": "uncountable",
}

# Priority order when the same sample_id appears in multiple directories.
# Lower index = higher priority (kept).
_STATUS_PRIORITY = ["countable", "empty", "uncountable"]


def load_agar_records(data_root: Path) -> list[ImageRecord]:
    """Walk *data_root* (which must contain ``countable/``, ``empty/``, and/or
    ``uncountable/`` sub-directories) and return one :class:`ImageRecord` per
    image found.

    When the same ``sample_id`` appears in more than one sub-directory (a known
    dataset quirk), the record from the highest-priority directory is kept and a
    warning is printed.  Priority order: ``countable`` > ``empty`` >
    ``uncountable``.

    Args:
        data_root: Path to the AGAR dataset root that contains sub-directories
            named ``countable``, ``empty``, and ``uncountable``.

    Returns:
        A list of validated :class:`ImageRecord` objects, one per image, sorted
        by ``sample_id``.

    Raises:
        DatasetFormatError: If any sidecar JSON is malformed, the referenced
            image file is missing or unreadable, or box coordinates fall
            outside the image dimensions.
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise DatasetFormatError(f"AGAR data root not found or not a directory: {data_root}")

    # First pass: collect all (sample_id → (status_priority, status_dir, json_path)) entries
    # so we can resolve cross-directory duplicates before doing expensive image I/O.
    candidates: dict[int, tuple[int, str, Path]] = {}  # sample_id → (priority, status, json_path)
    duplicates_found: list[str] = []

    for status_dir_name in _STATUS_PRIORITY:
        status_dir = data_root / status_dir_name
        if not status_dir.is_dir():
            continue
        priority = _STATUS_PRIORITY.index(status_dir_name)

        for json_path in sorted(status_dir.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                sample_id = int(payload["sample_id"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise DatasetFormatError(
                    f"Cannot read sample_id from {json_path}: {exc}"
                ) from exc

            if sample_id in candidates:
                existing_priority, existing_status, existing_path = candidates[sample_id]
                if priority < existing_priority:
                    # New record has HIGHER priority — replace the existing one.
                    duplicates_found.append(
                        f"sample_id={sample_id}: keeping {json_path} "
                        f"('{status_dir_name}'), dropping {existing_path} ('{existing_status}')"
                    )
                    candidates[sample_id] = (priority, status_dir_name, json_path)
                else:
                    # Existing record has higher or equal priority — keep it.
                    duplicates_found.append(
                        f"sample_id={sample_id}: keeping {existing_path} "
                        f"('{existing_status}'), dropping {json_path} ('{status_dir_name}')"
                    )
            else:
                candidates[sample_id] = (priority, status_dir_name, json_path)

    if duplicates_found:
        import warnings
        for msg in duplicates_found:
            warnings.warn(f"AGAR duplicate {msg}", stacklevel=2)

    # Second pass: build validated ImageRecord objects from the winning candidates.
    records: list[ImageRecord] = []
    file_hashes: dict[Path, str] = {}

    for sample_id in sorted(candidates):
        _, status_dir_name, json_path = candidates[sample_id]
        status_label = _STATUS_MAP[status_dir_name]

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetFormatError(f"Cannot read AGAR JSON {json_path}: {exc}") from exc

        try:
            colonies_number: int = int(payload["colonies_number"])
            raw_labels: list = payload.get("labels", []) or []
            background: str = str(payload.get("background", ""))
            classes: list = payload.get("classes", []) or []
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetFormatError(
                f"Missing or invalid field in {json_path}: {exc}"
            ) from exc

        # Locate the matching image file
        image_path = json_path.with_suffix(".jpg")
        if not image_path.is_file():
            image_path = json_path.with_suffix(".png")
        if not image_path.is_file():
            raise DatasetFormatError(
                f"No image file found for sidecar {json_path} (tried .jpg and .png)"
            )

        # Decode image dimensions
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except OSError as exc:
            raise DatasetFormatError(f"Cannot decode image {image_path}: {exc}") from exc

        # Build category name → stable int id mapping
        category_map: dict[str, int] = {}
        for cls_name in classes:
            if cls_name not in category_map:
                category_map[cls_name] = len(category_map) + 1

        # Parse bounding boxes
        boxes: list[Box] = []
        for label in raw_labels:
            try:
                cls_name = str(label["class"])
                bx = float(label["x"])
                by = float(label["y"])
                bw = float(label["width"])
                bh = float(label["height"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetFormatError(
                    f"Invalid label entry in {json_path}: {label!r} → {exc}"
                ) from exc

            if bw <= 0 or bh <= 0:
                raise DatasetFormatError(
                    f"Non-positive box dimensions in {json_path}: w={bw}, h={bh}"
                )
            # Clip boxes that marginally exceed image bounds (common in AGAR)
            bx = max(0.0, bx)
            by = max(0.0, by)
            bw = min(bw, img_w - bx)
            bh = min(bh, img_h - by)
            if bw <= 0 or bh <= 0:
                raise DatasetFormatError(
                    f"Box completely outside image bounds in {json_path}: "
                    f"x={bx}, y={by}, w={bw}, h={bh}, image={img_w}x{img_h}"
                )

            cat_id = category_map.setdefault(cls_name, len(category_map) + 1)
            boxes.append(Box(bx, by, bw, bh, cat_id, cls_name))

        # Hash each image file once
        if image_path not in file_hashes:
            file_hashes[image_path] = _sha256_file(image_path)

        records.append(
            ImageRecord(
                image_id=sample_id,
                image_path=(image_path.relative_to(data_root.parent)).as_posix(),
                width=img_w,
                height=img_h,
                boxes=tuple(boxes),
                group_id=str(sample_id),
                lighting=background if background else None,
                camera=None,
                status=status_label,
                sha256=file_hashes[image_path],
            )
        )

    if not records:
        raise DatasetFormatError(
            f"No AGAR image/JSON pairs found under {data_root}. "
            f"Expected sub-directories: countable/, empty/, uncountable/"
        )

    return records
