"""Stratified score reports and traceable visual error analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from colony_counting.evaluation import PredictionRecord, counting_metrics

DEFAULT_COUNT_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("empty", 0, 0),
    ("1-20", 1, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("over-100", 101, None),
)


def stratified_count_metrics(
    records: list[PredictionRecord],
    metadata_by_id: dict[str, dict[str, Any]],
    *,
    fields: tuple[str, ...] = ("lighting", "status", "camera", "culture_composition"),
) -> dict[str, Any]:
    """Report count metrics by fixed GT count bins and observed manifest metadata."""

    prediction_ids = {record.image_id for record in records}
    missing = sorted(prediction_ids - metadata_by_id.keys())
    if missing:
        raise ValueError(f"Missing metadata for prediction image IDs: {missing[:10]}.")
    result: dict[str, Any] = {"count_bins": {}, "metadata_strata": {}}
    for label, low, high in DEFAULT_COUNT_BINS:
        subset = [
            record
            for record in records
            if record.gt_count >= low and (high is None or record.gt_count <= high)
        ]
        if subset:
            result["count_bins"][label] = counting_metrics(subset)
    for field in fields:
        values: dict[str, list[PredictionRecord]] = defaultdict(list)
        for record in records:
            raw = metadata_by_id[record.image_id].get(field)
            value = str(raw) if raw not in (None, "") else "unspecified"
            values[value].append(record)
        if any(value != "unspecified" for value in values):
            result["metadata_strata"][field] = {
                value: counting_metrics(subset) for value, subset in sorted(values.items())
            }
    return result


def write_error_analysis(
    records: list[PredictionRecord],
    metadata_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    images_root: Path | None = None,
    gallery_size: int = 12,
) -> dict[str, Any]:
    """Write score strata, count plots, and an automatically ranked overlay gallery."""

    if not records:
        raise ValueError("Cannot analyze an empty prediction set.")
    if gallery_size < 1:
        raise ValueError("gallery_size must be positive.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = [record.predicted_count - record.gt_count for record in records]
    report = {
        "status": "computed_from_saved_predictions",
        "n_images": len(records),
        "overall_counting": counting_metrics(records),
        "stratified_counting": stratified_count_metrics(records, metadata_by_id),
        "worst_count_errors": [
            {
                "image_id": record.image_id,
                "gt_count": record.gt_count,
                "predicted_count": record.predicted_count,
                "signed_error": record.predicted_count - record.gt_count,
                "absolute_error": abs(record.predicted_count - record.gt_count),
            }
            for record in sorted(
                records,
                key=lambda item: (
                    -abs(item.predicted_count - item.gt_count),
                    item.image_id,
                ),
            )[:gallery_size]
        ],
        "gallery_status": "not_requested" if images_root is None else "written",
        "human_failure_labels": "not collected; gallery ranking is automatic",
    }
    if images_root is not None:
        report["gallery_status"] = _write_overlay_gallery(
            records, metadata_by_id, Path(images_root), output_dir, gallery_size
        )
    _write_json(report, output_dir / "analysis.json")
    _write_count_figures(records, errors, output_dir)
    return report


def _write_overlay_gallery(
    records: list[PredictionRecord],
    metadata_by_id: dict[str, dict[str, Any]],
    images_root: Path,
    output_dir: Path,
    gallery_size: int,
) -> str:
    import cv2

    gallery = output_dir / "worst_cases"
    gallery.mkdir(parents=True, exist_ok=True)
    selected = sorted(
        records,
        key=lambda item: (-abs(item.predicted_count - item.gt_count), item.image_id),
    )[:gallery_size]
    written = 0
    for rank, record in enumerate(selected, start=1):
        raw_path = metadata_by_id[record.image_id].get("image_path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        image_path = path if path.is_absolute() else images_root / path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        for box in record.gt_boxes:
            _draw_box(cv2, image, box, (40, 200, 40))
        for box in record.predicted_boxes:
            _draw_box(cv2, image, box, (30, 40, 240))
        cv2.putText(
            image,
            f"{record.image_id}: GT {record.gt_count} / pred {record.predicted_count}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )
        destination = gallery / f"{rank:02d}_{_safe_name(record.image_id)}.jpg"
        if cv2.imwrite(str(destination), image):
            written += 1
    return f"{written}/{len(selected)} automatically ranked images rendered"


def _draw_box(cv2: Any, image: np.ndarray, box: Any, color: tuple[int, int, int]) -> None:
    first = (round(box.x), round(box.y))
    second = (round(box.x + box.width), round(box.y + box.height))
    cv2.rectangle(image, first, second, color, 2)


def _write_count_figures(
    records: list[PredictionRecord], errors: list[int], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual = np.asarray([item.gt_count for item in records], dtype=np.float64)
    predicted = np.asarray([item.predicted_count for item in records], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].scatter(actual, predicted, alpha=0.7)
    extent = max(float(actual.max()), float(predicted.max()), 1.0)
    axes[0].plot([0, extent], [0, extent], color="black", linestyle="--")
    axes[0].set(xlabel="Ground-truth count", ylabel="Predicted count", title="Count agreement")
    mean = (actual + predicted) / 2
    axes[1].scatter(mean, errors, alpha=0.7)
    axes[1].axhline(float(np.mean(errors)), color="black", linestyle="--", label="Mean bias")
    axes[1].set(
        xlabel="Mean of GT and prediction",
        ylabel="Prediction minus GT",
        title="Bland-Altman count errors",
    )
    axes[1].legend()
    figure.savefig(output_dir / "count_error_figures.png", dpi=160)
    plt.close(figure)


def _write_json(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )
    return cleaned[:80] or "image"
