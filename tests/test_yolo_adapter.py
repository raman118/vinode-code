"""Ultralytics adapter contract tests with a fake predictor, not model performance evidence."""

import sys
import types

import numpy as np
import pytest

from colony_counting.detection.yolo import (
    _UltralyticsPredictor,
    infer_yolo,
    save_inference_result,
    train_yolo,
)


class _TensorArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    xyxy = _TensorArray([[2, 3, 12, 13]])
    conf = _TensorArray([0.75])
    cls = _TensorArray([2])

    def __len__(self):
        return 1


class _Result:
    boxes = _Boxes()


class _FakeModel:
    def __init__(self):
        self.arguments = None

    def predict(self, **kwargs):
        self.arguments = kwargs
        return [_Result()]


def test_adapter_extracts_ultralytics_boxes_and_forwards_thresholds():
    model = _FakeModel()
    predictor = _UltralyticsPredictor(
        model, image_size=640, confidence=0.3, device="cpu", class_agnostic_nms=True
    )
    detections = predictor(np.zeros((20, 20, 3), dtype=np.uint8))
    assert detections[0].x1 == 2
    assert detections[0].confidence == 0.75
    assert detections[0].category_id == 2
    assert model.arguments["agnostic_nms"] is True
    assert model.arguments["conf"] == 0.3


def test_missing_checkpoint_fails_before_model_import(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        infer_yolo(image=np.zeros((10, 10, 3), dtype=np.uint8), checkpoint=tmp_path / "missing.pt")


def test_inference_artifact_contains_geometry_hash_and_measured_latency(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture checkpoint bytes")
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    result_path, overlay_path = save_inference_result(
        image,
        [],
        checkpoint=checkpoint,
        output_dir=tmp_path / "prediction",
        inference_seconds=0.125,
        tiled=True,
        confidence=0.3,
        tile_size=512,
        overlap=0.2,
        nms_iou=0.5,
        device="cpu",
    )
    import json

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["count"] == 0
    assert payload["inference_seconds"] == 0.125
    assert payload["tile_size"] == 512
    assert len(payload["checkpoint_sha256"]) == 64
    assert overlay_path.is_file()


def test_training_wrapper_records_local_architecture_and_checkpoint_provenance(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True)
    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {dataset_root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: colony\n",
        encoding="utf-8",
    )
    weights = tmp_path / "random.pt"
    weights.write_bytes(b"synthetic weights")
    save_dir = tmp_path / "train-run"

    class FakeModel:
        model = types.SimpleNamespace(yaml={"yaml_file": "yolo11n.yaml"})
        trainer = types.SimpleNamespace(save_dir=save_dir)

        def train(self, **kwargs):
            self.arguments = kwargs

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.__version__ = "test-double"
    fake_ultralytics.YOLO = lambda _weights: FakeModel()
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "test-double"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    metadata_path = train_yolo(
        dataset_yaml=dataset_yaml,
        weights=weights,
        output_dir=tmp_path / "runs",
        epochs=1,
    )
    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "COMPLETED"
    assert metadata["model"]["architecture"] == "yolo11n.yaml"
    assert metadata["model"]["initialization_checkpoint_sha256"]
