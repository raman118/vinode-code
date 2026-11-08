"""Demo callback tests; the missing-checkpoint path never substitutes mock inference."""

from pathlib import Path

import numpy as np

from colony_counting.app import process_image


def test_yolo_method_reports_missing_checkpoint_without_fake_output():
    image = np.full((80, 80, 3), 230, dtype=np.uint8)
    classical, deep, counts, latency, error = process_image(
        image, "YOLO", 0.25, checkpoint=Path("not-a-real-checkpoint.pt")
    )
    assert classical is None
    assert deep is None
    assert "checkpoint required" in counts
    assert "not measured" in latency
    assert "Train a detector" in error


def test_classical_method_returns_real_overlay_and_count():
    image = np.full((80, 80, 3), 225, dtype=np.uint8)
    image[35:45, 35:45] = 20
    classical, deep, counts, latency, error = process_image(image, "Classical", 0.25)
    assert classical is not None and classical.shape == image.shape
    assert deep is None
    assert counts.startswith("Classical:")
    assert "s" in latency
    assert error == "No errors."
