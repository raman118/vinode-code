"""Known-answer evaluation tests using toy records, not benchmark evidence."""

import json

import pytest

from colony_counting.evaluation import (
    EvalBox,
    MatchCriterion,
    PredictionRecord,
    counting_metrics,
    detection_map_metrics,
    detection_metrics,
    match_boxes,
    paired_bootstrap_mae_difference,
    paired_wilcoxon_mae,
    read_predictions,
    select_detection_thresholds,
    write_predictions,
)


def _records(predictions):
    return [
        PredictionRecord("empty", 0, predictions[0]),
        PredictionRecord("a", 10, predictions[1]),
        PredictionRecord("b", 20, predictions[2]),
    ]


def test_counting_metrics_known_values_and_zero_gt_behavior():
    metrics = counting_metrics(_records([2, 12, 15]))
    assert metrics["mae"] == pytest.approx(3.0)
    assert metrics["rmse"] == pytest.approx((11) ** 0.5)
    assert metrics["mean_signed_error"] == pytest.approx(-1 / 3)
    assert metrics["mape_percent_nonzero_gt"] == pytest.approx(22.5)
    assert metrics["mape_n_images"] == 2
    assert metrics["absolute_tolerance_accuracy"] == 1.0
    assert metrics["relative_tolerance_accuracy_nonzero_gt"] == pytest.approx(0.0)
    assert metrics["empty_false_positive_image_rate"] == 1.0
    assert metrics["empty_mean_false_positive_count"] == 2


def test_undefined_metrics_are_explicit():
    metrics = counting_metrics([PredictionRecord("empty", 0, 0)])
    assert metrics["mape_percent_nonzero_gt"] is None
    assert metrics["r2"] is None
    assert metrics["pearson_r"] is None
    assert metrics["relative_tolerance_accuracy_nonzero_gt"] is None


def test_matching_maximizes_valid_matches_before_iou_quality():
    gt = [EvalBox(0, 0, 10, 10), EvalBox(10, 0, 10, 10)]
    pred = [EvalBox(5, 0, 10, 10), EvalBox(0, 0, 10, 10)]
    pairs = match_boxes(gt, pred, MatchCriterion(mode="iou", threshold=0.3))
    assert len(pairs) == 2
    record = PredictionRecord("p", 2, 2, tuple(gt), tuple(pred))
    metrics = detection_metrics([record], MatchCriterion(mode="iou", threshold=0.3))
    assert metrics["tp"] == 2
    assert metrics["fp"] == metrics["fn"] == 0
    assert metrics["f1"] == 1


def test_one_prediction_cannot_match_two_ground_truth_boxes():
    gt = (EvalBox(0, 0, 10, 10), EvalBox(1, 0, 10, 10))
    pred = (EvalBox(0, 0, 10, 10),)
    assert len(match_boxes(gt, pred, MatchCriterion())) == 1


def test_category_matching_is_optional_and_explicit():
    gt = (EvalBox(0, 0, 10, 10, "a"),)
    pred = (EvalBox(0, 0, 10, 10, "b"),)
    assert len(match_boxes(gt, pred, MatchCriterion())) == 1
    assert len(match_boxes(gt, pred, MatchCriterion(require_category_match=True))) == 0


def test_detection_evaluation_checks_box_count_consistency():
    with pytest.raises(ValueError, match="box count mismatch"):
        detection_metrics([PredictionRecord("img", 1, 1)], MatchCriterion())


def test_prediction_jsonl_round_trip(tmp_path):
    records = [
        PredictionRecord("img", 1, 1, (EvalBox(1, 2, 3, 4, "a"),), metadata={"lighting": "ring"})
    ]
    output = tmp_path / "predictions.jsonl"
    write_predictions(records, output)
    assert read_predictions(output) == records
    assert json.loads(output.read_text(encoding="utf-8"))["image_id"] == "img"


def test_prediction_reader_rejects_fractional_counts(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"image_id":"i","gt_count":1.5,"predicted_count":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative integer"):
        read_predictions(path)


def test_paired_bootstrap_is_seeded_and_checks_image_sets():
    first = _records([0, 11, 20])
    second = _records([1, 10, 20])
    result = paired_bootstrap_mae_difference(first, second, resamples=2000, seed=7)
    assert result == paired_bootstrap_mae_difference(first, second, resamples=2000, seed=7)
    assert result["n_images"] == 3
    with pytest.raises(ValueError, match="image sets differ"):
        paired_bootstrap_mae_difference(first, second[:-1])


def test_wilcoxon_handles_all_zero_and_computes_nonzero_pairs():
    first = _records([0, 11, 20])
    identical = _records([0, 11, 20])
    all_zero = paired_wilcoxon_mae(first, identical)
    assert all_zero["p_value"] == 1.0
    assert all_zero["n_nonzero_differences"] == 0
    second = _records([1, 13, 18])
    result = paired_wilcoxon_mae(first, second)
    assert result["n_nonzero_differences"] == 3
    assert 0 <= result["p_value"] <= 1
    assert -1 <= result["rank_biserial_effect"] <= 1


def test_threshold_selection_uses_confidence_and_reports_separate_objectives():
    gt = (EvalBox(0, 0, 10, 10),)
    predictions = (
        EvalBox(0, 0, 10, 10, confidence=0.8),
        EvalBox(20, 20, 8, 8, confidence=0.6),
    )
    record = PredictionRecord("val-image", 1, 2, gt, predictions)
    result = select_detection_thresholds(
        [record], [0.5, 0.7, 0.9], MatchCriterion(mode="iou", threshold=0.5)
    )
    assert result["threshold_count_mae"] == 0.7
    assert result["threshold_detection_f1"] == 0.7
    assert result["count_mae_at_selected_threshold"] == 0
    assert result["detection_f1_at_selected_threshold"] == 1
    assert len(result["threshold_curve"]) == 3


def test_threshold_selection_requires_scores_for_all_predictions():
    record = PredictionRecord("val-image", 1, 1, (EvalBox(0, 0, 10, 10),), (EvalBox(0, 0, 10, 10),))
    with pytest.raises(ValueError, match="confidence score"):
        select_detection_thresholds([record], [0.5], MatchCriterion())


def test_detection_map_uses_confidence_ranked_interpolated_ap():
    first = PredictionRecord(
        "image-a",
        1,
        2,
        (EvalBox(0, 0, 10, 10, "species-a"),),
        (
            EvalBox(20, 20, 8, 8, "species-a", 0.95),
            EvalBox(0, 0, 10, 10, "species-a", 0.80),
        ),
    )
    second = PredictionRecord(
        "image-b",
        1,
        1,
        (EvalBox(5, 5, 10, 10, "species-a"),),
        (EvalBox(5, 5, 10, 10, "species-a", 0.70),),
    )
    result = detection_map_metrics([first, second], class_agnostic=False)
    assert result["map_50"] == pytest.approx(2 / 3, abs=0.01)
    assert result["map_50_95"] == pytest.approx(2 / 3, abs=0.01)
    assert result["per_iou"]["0.50"]["class_ap"]["species-a"] == pytest.approx(2 / 3, abs=0.01)


def test_detection_map_requires_confidence_for_prediction_boxes():
    record = PredictionRecord(
        "image",
        1,
        1,
        (EvalBox(0, 0, 10, 10),),
        (EvalBox(0, 0, 10, 10),),
    )
    with pytest.raises(ValueError, match="confidence"):
        detection_map_metrics([record])
