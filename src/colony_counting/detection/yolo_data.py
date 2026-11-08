"""Convert validated COCO-normalized records into Ultralytics YOLO dataset files."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from colony_counting.data.coco import ImageRecord


def export_yolo_dataset(
    records: list[ImageRecord],
    image_root: Path,
    output_dir: Path,
    *,
    class_agnostic: bool = True,
) -> Path:
    """Copy split images and emit YOLO normalized labels, including empty label files.

    Records without a frozen ``train``, ``val``, or ``test`` split are rejected. The source status
    policy remains an explicit upstream responsibility until the AGAR label vocabulary is known.
    """

    image_root, output_dir = Path(image_root), Path(output_dir)
    if not records:
        raise ValueError("Cannot export an empty dataset.")
    allowed_splits = {"train", "val", "test"}
    for record in records:
        if record.split not in allowed_splits:
            raise ValueError(
                f"Image {record.image_id} has no valid frozen split: {record.split!r}."
            )

    seen_paths: set[str] = set()
    for record in records:
        key = record.image_path.casefold()
        if key in seen_paths:
            raise ValueError(
                f"Duplicate source image path across dataset splits: {record.image_path}"
            )
        seen_paths.add(key)

    category_ids = sorted({box.category_id for record in records for box in record.boxes})
    if not category_ids:
        raise ValueError("YOLO export requires at least one annotated object category.")
    if class_agnostic:
        class_ids = {category_id: 0 for category_id in category_ids}
        names = {0: "colony"}
    else:
        class_ids = {category_id: index for index, category_id in enumerate(category_ids)}
        category_names: dict[int, str] = {}
        for record in records:
            for box in record.boxes:
                existing = category_names.setdefault(box.category_id, box.category_name)
                if existing != box.category_name:
                    raise ValueError(f"Category id {box.category_id} has inconsistent names.")
        names = {
            class_ids[category_id]: category_names[category_id] for category_id in category_ids
        }

    for record in records:
        source = image_root / Path(record.image_path)
        if not source.is_file():
            raise FileNotFoundError(f"Image file is missing during YOLO export: {source}")
        relative_path = Path(record.image_path)
        target_image = output_dir / "images" / str(record.split) / relative_path
        target_label = output_dir / "labels" / str(record.split) / relative_path.with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_image)
        lines = []
        for box in record.boxes:
            center_x = (box.x + box.width / 2) / record.width
            center_y = (box.y + box.height / 2) / record.height
            normalized_width = box.width / record.width
            normalized_height = box.height / record.height
            if not (
                0 <= center_x <= 1
                and 0 <= center_y <= 1
                and 0 < normalized_width <= 1
                and 0 < normalized_height <= 1
            ):
                raise ValueError(f"Box cannot be represented as normalized YOLO coordinates: {box}")
            lines.append(
                f"{class_ids[box.category_id]} {center_x:.8f} {center_y:.8f} "
                f"{normalized_width:.8f} {normalized_height:.8f}"
            )
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_yaml = output_dir / "dataset.yaml"
    payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    dataset_yaml.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return dataset_yaml
