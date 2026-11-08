"""EDA tests use synthetic records and validate summaries, not biological findings."""

from colony_counting.data.coco import Box, ImageRecord
from colony_counting.data.eda import summarize_records, write_eda


def test_summary_keeps_zero_boxes_distinct_from_status():
    records = [
        ImageRecord(
            1,
            "a.png",
            100,
            100,
            (Box(2, 3, 4, 5, 1, "species-a"),),
            "p1",
            "ring",
            "cam",
            "countable",
        ),
        ImageRecord(2, "b.png", 80, 80, (), "p2", None, None, None),
    ]
    summary = summarize_records(records)
    assert summary["image_count"] == 2
    assert summary["annotated_object_count"] == 1
    assert summary["images_with_zero_boxes"] == 1
    assert summary["status_distribution"] == {"countable": 1}
    assert summary["exact_duplicate_image_groups"] == 0
    assert "not assumed to be empty plates" in summary["interpretation_note"]


def test_write_eda_creates_summary_and_plots(tmp_path):
    records = [ImageRecord(1, "a.png", 100, 100, (), "p1", None, None, "empty")]
    summary = write_eda(records, tmp_path)
    assert summary["image_count"] == 1
    assert (tmp_path / "eda_summary.json").is_file()
    assert (tmp_path / "eda_distributions.png").is_file()
    assert (tmp_path / "eda_box_sizes.png").is_file()
