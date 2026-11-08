"""Model-independent counting, detection, and paired statistical evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class EvalBox:
    """Absolute image-coordinate XYWH box with an optional category label."""

    x: float
    y: float
    width: float
    height: float
    category: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class PredictionRecord:
    """Ground truth and predictions for one image in a method-independent format."""

    image_id: str
    gt_count: int
    predicted_count: int
    gt_boxes: tuple[EvalBox, ...] = ()
    predicted_boxes: tuple[EvalBox, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchCriterion:
    """Detection matching mode and threshold, supplied from the project protocol."""

    mode: Literal["iou", "centroid"] = "iou"
    threshold: float = 0.5
    require_category_match: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"iou", "centroid"}:
            raise ValueError("Matching mode must be 'iou' or 'centroid'.")
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("Matching threshold must be positive and finite.")
        if self.mode == "iou" and self.threshold > 1:
            raise ValueError("IoU threshold must be <= 1.")


def validate_prediction_records(records: list[PredictionRecord]) -> None:
    """Validate image IDs, count fields, and box geometry."""

    seen: set[str] = set()
    for record in records:
        if not isinstance(record.image_id, str) or not record.image_id or record.image_id in seen:
            raise ValueError(f"Image IDs must be non-empty and unique; found {record.image_id!r}.")
        seen.add(record.image_id)
        for label, count in (
            ("gt_count", record.gt_count),
            ("predicted_count", record.predicted_count),
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{label} must be a non-negative integer for {record.image_id}.")
        for box in (*record.gt_boxes, *record.predicted_boxes):
            if not all(math.isfinite(value) for value in (box.x, box.y, box.width, box.height)):
                raise ValueError(f"Non-finite box in image {record.image_id}.")
            if box.x < 0 or box.y < 0 or box.width <= 0 or box.height <= 0:
                raise ValueError(f"Invalid box geometry in image {record.image_id}: {box}.")
            if box.confidence is not None and (
                not math.isfinite(box.confidence) or not 0 <= box.confidence <= 1
            ):
                raise ValueError(f"Box confidence must be finite and in [0, 1]: {box}.")


def write_predictions(records: list[PredictionRecord], output_file: Path) -> None:
    """Atomically write validated predictions as deterministic JSONL."""

    validate_prediction_records(records)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with temp_file.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            row = asdict(record)
            row["gt_boxes"] = [asdict(box) for box in record.gt_boxes]
            row["predicted_boxes"] = [asdict(box) for box in record.predicted_boxes]
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
    temp_file.replace(output_file)


def read_predictions(input_file: Path) -> list[PredictionRecord]:
    """Read and validate model-independent prediction JSONL."""

    records: list[PredictionRecord] = []
    for line_number, line in enumerate(
        Path(input_file).read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("record must be an object")
            if not isinstance(row.get("image_id"), str):
                raise TypeError("image_id must be a string")
            records.append(
                PredictionRecord(
                    image_id=row["image_id"],
                    gt_count=row["gt_count"],
                    predicted_count=row["predicted_count"],
                    gt_boxes=tuple(EvalBox(**box) for box in row.get("gt_boxes", [])),
                    predicted_boxes=tuple(EvalBox(**box) for box in row.get("predicted_boxes", [])),
                    metadata=row.get("metadata", {}),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid prediction JSONL at {input_file}:{line_number}: {exc}"
            ) from exc
    validate_prediction_records(records)
    return records


def counting_metrics(
    records: list[PredictionRecord],
    *,
    absolute_tolerance: int = 5,
    relative_tolerance: float = 0.10,
) -> dict[str, float | int | None]:
    """Compute count metrics and report undefined values as ``None``."""

    validate_prediction_records(records)
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("Count tolerances cannot be negative.")
    if not records:
        raise ValueError("Cannot evaluate an empty prediction set.")
    gt = np.asarray([record.gt_count for record in records], dtype=np.float64)
    pred = np.asarray([record.predicted_count for record in records], dtype=np.float64)
    errors = pred - gt
    abs_errors = np.abs(errors)
    nonzero = gt != 0
    empty = ~nonzero
    if len(records) > 1 and np.var(gt) > 0:
        r2: float | None = float(1 - np.sum(errors**2) / np.sum((gt - gt.mean()) ** 2))
        pearson: float | None = float(np.corrcoef(gt, pred)[0, 1]) if np.var(pred) > 0 else None
    else:
        r2, pearson = None, None
    return {
        "n_images": len(records),
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_signed_error": float(errors.mean()),
        "mape_percent_nonzero_gt": float(np.mean(abs_errors[nonzero] / gt[nonzero]) * 100)
        if nonzero.any()
        else None,
        "mape_n_images": int(nonzero.sum()),
        "r2": r2,
        "pearson_r": pearson,
        "absolute_tolerance": absolute_tolerance,
        "absolute_tolerance_accuracy": float(np.mean(abs_errors <= absolute_tolerance)),
        "relative_tolerance": relative_tolerance,
        "relative_tolerance_accuracy_nonzero_gt": (
            float(np.mean(abs_errors[nonzero] <= relative_tolerance * gt[nonzero]))
            if nonzero.any()
            else None
        ),
        "relative_tolerance_n_images": int(nonzero.sum()),
        "n_empty_gt": int(empty.sum()),
        "empty_false_positive_image_rate": float(np.mean(pred[empty] > 0)) if empty.any() else None,
        "empty_mean_false_positive_count": float(pred[empty].mean()) if empty.any() else None,
        "empty_exact_accuracy": float(np.mean(pred[empty] == 0)) if empty.any() else None,
    }


def detection_metrics(
    records: list[PredictionRecord], criterion: MatchCriterion
) -> dict[str, float | int | None]:
    """Compute one-to-one aggregate precision/recall/F1 for saved boxes."""

    validate_prediction_records(records)
    for record in records:
        if record.gt_count != len(record.gt_boxes):
            raise ValueError(
                f"Ground-truth box count mismatch for detection evaluation of {record.image_id}."
            )
        if record.predicted_count != len(record.predicted_boxes):
            raise ValueError(
                f"Prediction box count mismatch for detection evaluation of {record.image_id}."
            )
    true_positive = false_positive = false_negative = 0
    for record in records:
        pairs = match_boxes(record.gt_boxes, record.predicted_boxes, criterion)
        true_positive += len(pairs)
        false_positive += len(record.predicted_boxes) - len(pairs)
        false_negative += len(record.gt_boxes) - len(pairs)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    )
    f1 = (
        (2 * precision * recall / (precision + recall))
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matching_mode": criterion.mode,
        "matching_threshold": criterion.threshold,
        "require_category_match": criterion.require_category_match,
    }


def detection_map_metrics(
    records: list[PredictionRecord],
    *,
    iou_thresholds: list[float] | None = None,
    class_agnostic: bool = True,
) -> dict[str, Any]:
    """Compute 101-point interpolated AP using confidence-ranked one-to-one detections.

    This follows the common object-detection AP convention: predictions are matched greedily in
    descending confidence order per image and category. It is distinct from the maximum-cardinality
    assignment used by :func:`detection_metrics` for a single operating point.
    """

    validate_prediction_records(records)
    if not records:
        raise ValueError("Cannot compute detection AP on an empty prediction set.")
    for record in records:
        if record.gt_count != len(record.gt_boxes):
            raise ValueError(f"Ground-truth box count mismatch for {record.image_id}.")
        if record.predicted_count != len(record.predicted_boxes):
            raise ValueError(f"Prediction box count mismatch for {record.image_id}.")
        if any(box.confidence is None for box in record.predicted_boxes):
            raise ValueError("Every predicted box must have confidence for AP evaluation.")
    thresholds = iou_thresholds or [0.5 + 0.05 * index for index in range(10)]
    if not thresholds or any(
        not math.isfinite(value) or not 0 < value <= 1 for value in thresholds
    ):
        raise ValueError("IoU thresholds must be non-empty finite values in (0, 1].")
    if class_agnostic:
        categories = ["__all__"]
    else:
        categories = sorted({str(box.category) for record in records for box in record.gt_boxes})
    positives_by_class = {
        category: sum(
            1
            for record in records
            for box in record.gt_boxes
            if class_agnostic or str(box.category) == category
        )
        for category in categories
    }
    per_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        class_ap: dict[str, float | None] = {}
        for category in categories:
            n_ground_truth = positives_by_class[category]
            if n_ground_truth == 0:
                class_ap[category] = None
                continue
            ranked: list[tuple[float, str, int, EvalBox]] = []
            gt_by_image: dict[str, list[EvalBox]] = {}
            for record in records:
                gt_by_image[record.image_id] = [
                    box
                    for box in record.gt_boxes
                    if class_agnostic or str(box.category) == category
                ]
                for index, box in enumerate(record.predicted_boxes):
                    if class_agnostic or str(box.category) == category:
                        ranked.append((float(box.confidence), record.image_id, index, box))
            ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
            matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
            true_positive = np.zeros(len(ranked), dtype=np.float64)
            false_positive = np.zeros(len(ranked), dtype=np.float64)
            for rank, (_, image_id, _, predicted_box) in enumerate(ranked):
                targets = gt_by_image[image_id]
                candidates = sorted(
                    (
                        (_iou(predicted_box, gt_box), gt_index)
                        for gt_index, gt_box in enumerate(targets)
                        if gt_index not in matched[image_id]
                    ),
                    reverse=True,
                )
                if candidates and candidates[0][0] >= threshold:
                    matched[image_id].add(candidates[0][1])
                    true_positive[rank] = 1
                else:
                    false_positive[rank] = 1
            recall = np.cumsum(true_positive) / n_ground_truth
            precision = np.cumsum(true_positive) / np.maximum(
                np.cumsum(true_positive + false_positive), np.finfo(float).eps
            )
            class_ap[category] = _interpolated_ap(recall, precision)
        valid_ap = [value for value in class_ap.values() if value is not None]
        per_threshold[f"{threshold:.2f}"] = {
            "mean_ap": float(np.mean(valid_ap)) if valid_ap else None,
            "class_ap": class_ap,
            "n_classes_with_ground_truth": len(valid_ap),
        }
    map50 = per_threshold.get("0.50", {}).get("mean_ap")
    available = [item["mean_ap"] for item in per_threshold.values() if item["mean_ap"] is not None]
    return {
        "map_50": map50,
        "map_50_95": float(np.mean(available)) if available else None,
        "iou_thresholds": thresholds,
        "class_agnostic": class_agnostic,
        "interpolation_points": 101,
        "matching": "descending-confidence greedy one-to-one matching per image/category",
        "per_iou": per_threshold,
    }


def select_detection_thresholds(
    records: list[PredictionRecord],
    thresholds: list[float],
    criterion: MatchCriterion,
) -> dict[str, Any]:
    """Select validation thresholds separately for count MAE and detection F1."""

    validate_prediction_records(records)
    if not records:
        raise ValueError("Cannot tune a threshold on an empty prediction set.")
    if not thresholds:
        raise ValueError("At least one confidence threshold is required.")
    candidates = sorted({float(value) for value in thresholds})
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in candidates):
        raise ValueError("Confidence thresholds must be finite values in [0, 1].")
    if any(box.confidence is None for record in records for box in record.predicted_boxes):
        raise ValueError("Every predicted box must carry a confidence score for threshold tuning.")
    curve = []
    for threshold in candidates:
        filtered = [
            PredictionRecord(
                image_id=record.image_id,
                gt_count=record.gt_count,
                predicted_count=sum(box.confidence >= threshold for box in record.predicted_boxes),
                gt_boxes=record.gt_boxes,
                predicted_boxes=tuple(
                    box for box in record.predicted_boxes if box.confidence >= threshold
                ),
                metadata=record.metadata,
            )
            for record in records
        ]
        curve.append(
            {
                "threshold": threshold,
                "counting": counting_metrics(filtered),
                "detection": detection_metrics(filtered, criterion),
            }
        )
    best_count = min(
        curve,
        key=lambda row: (row["counting"]["mae"], -row["threshold"]),
    )
    best_f1 = min(
        curve,
        key=lambda row: (-row["detection"]["f1"], -row["threshold"]),
    )
    return {
        "selection_split": "caller_must_enforce_validation_split",
        "matching_criterion": {
            "mode": criterion.mode,
            "threshold": criterion.threshold,
            "require_category_match": criterion.require_category_match,
        },
        "threshold_count_mae": best_count["threshold"],
        "threshold_detection_f1": best_f1["threshold"],
        "count_mae_at_selected_threshold": best_count["counting"]["mae"],
        "detection_f1_at_selected_threshold": best_f1["detection"]["f1"],
        "threshold_curve": curve,
        "tie_break": "highest threshold among candidates tied on the selection metric",
    }


def match_boxes(
    gt_boxes: tuple[EvalBox, ...] | list[EvalBox],
    predicted_boxes: tuple[EvalBox, ...] | list[EvalBox],
    criterion: MatchCriterion,
) -> list[tuple[int, int]]:
    """Return valid one-to-one matches, maximizing valid match count before match quality."""

    if not gt_boxes or not predicted_boxes:
        return []
    quality = np.zeros((len(gt_boxes), len(predicted_boxes)), dtype=np.float64)
    valid = np.zeros_like(quality, dtype=bool)
    for gt_index, gt_box in enumerate(gt_boxes):
        for pred_index, predicted_box in enumerate(predicted_boxes):
            if criterion.require_category_match and (
                gt_box.category is None or gt_box.category != predicted_box.category
            ):
                continue
            if criterion.mode == "iou":
                overlap = _iou(gt_box, predicted_box)
                quality[gt_index, pred_index] = overlap
                valid[gt_index, pred_index] = overlap >= criterion.threshold
            else:
                distance = math.dist(
                    (gt_box.x + gt_box.width / 2, gt_box.y + gt_box.height / 2),
                    (
                        predicted_box.x + predicted_box.width / 2,
                        predicted_box.y + predicted_box.height / 2,
                    ),
                )
                quality[gt_index, pred_index] = 1 / (1 + distance / criterion.threshold)
                valid[gt_index, pred_index] = distance <= criterion.threshold

    from scipy.optimize import linear_sum_assignment

    cardinality_bound = min(len(gt_boxes), len(predicted_boxes))
    scores = np.where(valid, cardinality_bound + 1 + quality, 0.0)
    gt_indices, pred_indices = linear_sum_assignment(scores, maximize=True)
    return [
        (int(gt_index), int(pred_index))
        for gt_index, pred_index in zip(gt_indices, pred_indices, strict=True)
        if valid[gt_index, pred_index]
    ]


def paired_bootstrap_mae_difference(
    first: list[PredictionRecord],
    second: list[PredictionRecord],
    *,
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired image bootstrap CI for ``MAE(first) - MAE(second)``."""

    first, second = _align_pairs(first, second)
    if resamples < 1:
        raise ValueError("resamples must be positive")
    first_abs = np.abs(
        np.asarray([r.predicted_count - r.gt_count for r in first], dtype=np.float64)
    )
    second_abs = np.abs(
        np.asarray([r.predicted_count - r.gt_count for r in second], dtype=np.float64)
    )
    paired_difference = first_abs - second_abs
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired_difference), size=(resamples, len(paired_difference)))
    samples = paired_difference[indices].mean(axis=1)
    return {
        "n_images": len(first),
        "resamples": resamples,
        "seed": seed,
        "mae_difference_first_minus_second": float(paired_difference.mean()),
        "ci_95_low": float(np.quantile(samples, 0.025)),
        "ci_95_high": float(np.quantile(samples, 0.975)),
    }


def paired_group_bootstrap_mae_difference(
    first: list[PredictionRecord],
    second: list[PredictionRecord],
    groups: dict[str, str],
    *,
    resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Paired cluster bootstrap that resamples image groups, retaining within-group dependence."""

    first, second = _align_pairs(first, second)
    ids = [record.image_id for record in first]
    missing_groups = sorted(set(ids) - groups.keys())
    if missing_groups:
        raise ValueError(f"Missing group IDs for prediction images: {missing_groups[:10]}.")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    group_to_indices: dict[str, list[int]] = {}
    for index, image_id in enumerate(ids):
        group_to_indices.setdefault(str(groups[image_id]), []).append(index)
    group_names = sorted(group_to_indices)
    if not group_names:
        raise ValueError("At least one group is required for group bootstrap.")
    first_abs = np.asarray(
        [abs(record.predicted_count - record.gt_count) for record in first], dtype=np.float64
    )
    second_abs = np.asarray(
        [abs(record.predicted_count - record.gt_count) for record in second], dtype=np.float64
    )
    difference = first_abs - second_abs
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        chosen = rng.integers(0, len(group_names), size=len(group_names))
        image_indices = [
            index
            for group_index in chosen
            for index in group_to_indices[group_names[int(group_index)]]
        ]
        samples[sample_index] = difference[image_indices].mean()
    return {
        "n_images": len(first),
        "n_groups": len(group_names),
        "resamples": resamples,
        "seed": seed,
        "mae_difference_first_minus_second": float(difference.mean()),
        "ci_95_low": float(np.quantile(samples, 0.025)),
        "ci_95_high": float(np.quantile(samples, 0.975)),
        "resampling_unit": "group",
    }


def paired_wilcoxon_mae(
    first: list[PredictionRecord], second: list[PredictionRecord]
) -> dict[str, float | int | str]:
    """Paired Wilcoxon test on per-image absolute count errors with explicit zero handling."""

    first, second = _align_pairs(first, second)
    first_abs = np.abs(
        np.asarray([r.predicted_count - r.gt_count for r in first], dtype=np.float64)
    )
    second_abs = np.abs(
        np.asarray([r.predicted_count - r.gt_count for r in second], dtype=np.float64)
    )
    differences = first_abs - second_abs
    nonzero = differences != 0
    if not nonzero.any():
        return {
            "n_images": len(first),
            "n_nonzero_differences": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial_effect": 0.0,
            "status": "all paired differences are zero",
        }
    from scipy.stats import rankdata, wilcoxon

    active = differences[nonzero]
    ranks = rankdata(np.abs(active), method="average")
    positive_rank_sum = float(ranks[active > 0].sum())
    negative_rank_sum = float(ranks[active < 0].sum())
    total_rank_sum = positive_rank_sum + negative_rank_sum
    test = wilcoxon(active, zero_method="wilcox", alternative="two-sided", method="auto")
    return {
        "n_images": len(first),
        "n_nonzero_differences": int(nonzero.sum()),
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "rank_biserial_effect": (positive_rank_sum - negative_rank_sum) / total_rank_sum,
        "status": "computed; zero differences omitted; ties use average ranks",
    }


def _align_pairs(
    first: list[PredictionRecord], second: list[PredictionRecord]
) -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    validate_prediction_records(first)
    validate_prediction_records(second)
    first_map = {record.image_id: record for record in first}
    second_map = {record.image_id: record for record in second}
    if first_map.keys() != second_map.keys():
        missing_first = sorted(second_map.keys() - first_map.keys())
        missing_second = sorted(first_map.keys() - second_map.keys())
        raise ValueError(
            f"Prediction image sets differ; missing first={missing_first}, missing second={missing_second}."
        )
    ids = sorted(first_map)
    for image_id in ids:
        if first_map[image_id].gt_count != second_map[image_id].gt_count:
            raise ValueError(f"Ground-truth counts differ between methods for image {image_id}.")
        if first_map[image_id].gt_boxes != second_map[image_id].gt_boxes:
            raise ValueError(f"Ground-truth boxes differ between methods for image {image_id}.")
    return [first_map[key] for key in ids], [second_map[key] for key in ids]


def _iou(first: EvalBox, second: EvalBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.0


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Average the maximum observed precision at each of 101 recall values."""

    recall_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_samples = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(recall_envelope[recall >= sample].max()) if np.any(recall >= sample) else 0.0
        for sample in recall_samples
    ]
    return float(np.mean(interpolated))
