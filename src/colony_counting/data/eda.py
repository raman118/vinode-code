"""Exploratory summaries and diagnostic figures from validated manifest records."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from colony_counting.data.coco import ImageRecord


def summarize_records(records: list[ImageRecord]) -> dict[str, Any]:
    """Summarize observed manifest fields without inferring missing metadata."""

    species = Counter(box.category_name for record in records for box in record.boxes)
    status = Counter(record.status for record in records if record.status is not None)
    lighting = Counter(record.lighting for record in records if record.lighting is not None)
    camera = Counter(record.camera for record in records if record.camera is not None)
    resolutions = Counter(f"{record.width}x{record.height}" for record in records)
    counts = [record.count for record in records]
    content_ids: dict[str, list[str]] = {}
    for record in records:
        if record.sha256:
            content_ids.setdefault(record.sha256, []).append(str(record.image_id))
    duplicate_content = [sorted(ids) for ids in content_ids.values() if len(ids) > 1]
    box_areas = [box.width * box.height for record in records for box in record.boxes]
    box_area_fractions = [
        (box.width * box.height) / (record.width * record.height)
        for record in records
        for box in record.boxes
    ]
    busiest = sorted(records, key=lambda record: (-record.count, str(record.image_id)))[:20]
    return {
        "image_count": len(records),
        "exact_duplicate_image_groups": len(duplicate_content),
        "images_in_exact_duplicate_groups": sum(len(group) for group in duplicate_content),
        "exact_duplicate_image_ids": duplicate_content,
        "annotated_object_count": sum(counts),
        "annotation_count_distribution": dict(sorted(Counter(counts).items())),
        "images_with_zero_boxes": sum(count == 0 for count in counts),
        "status_distribution": dict(sorted(status.items())),
        "species_annotation_distribution": dict(sorted(species.items())),
        "lighting_distribution": dict(sorted(lighting.items())),
        "camera_distribution": dict(sorted(camera.items())),
        "resolution_distribution": dict(sorted(resolutions.items())),
        "box_area_pixels": _numeric_summary(box_areas),
        "box_area_fraction_of_image": _numeric_summary(box_area_fractions),
        "busiest_images_by_annotation_count": [
            {"image_id": record.image_id, "count": record.count, "path": record.image_path}
            for record in busiest
        ],
        "interpretation_note": (
            "Zero-box images are not assumed to be empty plates; consult source status labels. "
            "Absent metadata is omitted, not inferred."
        ),
    }


def write_eda(records: list[ImageRecord], output_dir: Path) -> dict[str, Any]:
    """Write JSON summary and diagnostic PNG figures from validated records."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_records(records)
    summary_path = output_dir / "eda_summary.json"
    temp_path = summary_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(summary_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("EDA plots require matplotlib; install the analysis extra.") from exc

    counts = [record.count for record in records]
    resolutions = Counter(f"{record.width}x{record.height}" for record in records)
    species = Counter(box.category_name for record in records for box in record.boxes)
    lighting = Counter(record.lighting or "unspecified" for record in records)
    box_area_fraction = [
        (box.width * box.height) / (record.width * record.height)
        for record in records
        for box in record.boxes
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].hist(counts, bins=min(40, max(1, len(set(counts)))))
    axes[0, 0].set(title="Annotations per image", xlabel="Box count", ylabel="Images")
    _bar(axes[0, 1], species, "Object category labels", "Category", "Annotations")
    _bar(axes[1, 0], lighting, "Lighting metadata", "Lighting", "Images")
    _bar(axes[1, 1], resolutions, "Image dimensions", "Width x height", "Images")
    figure.savefig(output_dir / "eda_distributions.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    if box_area_fraction:
        axis.hist(box_area_fraction, bins=40)
    axis.set(title="Box area fraction of image", xlabel="Box area / image area", ylabel="Boxes")
    figure.savefig(output_dir / "eda_box_sizes.png", dpi=160)
    plt.close(figure)
    return summary


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def _bar(axis: Any, counts: Counter[str], title: str, xlabel: str, ylabel: str) -> None:
    ordered = sorted(counts.items(), key=lambda pair: pair[0])
    axis.bar([item[0] for item in ordered], [item[1] for item in ordered])
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.tick_params(axis="x", labelrotation=35)
