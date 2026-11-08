"""Optional Ultralytics training and tiled-inference adapters."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from colony_counting.detection.tiling import (
    DetectionBox,
    filter_plate_exterior,
    infer_tiled,
)


def train_yolo(
    *,
    dataset_yaml: Path,
    weights: Path,
    output_dir: Path,
    epochs: int = 50,
    image_size: int = 640,
    batch_size: int = 4,
    patience: int = 20,
    seed: int = 42,
    workers: int = 2,
    device: str = "auto",
    run_name: str = "colony-yolo",
) -> Path:
    """Train from an explicit local checkpoint and persist experiment metadata."""

    dataset_yaml, weights, output_dir = Path(dataset_yaml), Path(weights), Path(output_dir)
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"YOLO dataset YAML does not exist: {dataset_yaml}")
    if not weights.is_file():
        raise FileNotFoundError(
            f"Local YOLO weights do not exist: {weights}. Provide an authorized checkpoint explicitly."
        )
    if epochs < 1 or image_size < 32 or batch_size < 1 or patience < 0 or workers < 0:
        raise ValueError(
            "epochs, image size, batch size, patience, or worker settings are invalid."
        )
    data_config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(data_config, dict) or not all(
        key in data_config for key in ("train", "val", "names")
    ):
        raise ValueError("Dataset YAML must define train, val, and names fields.")
    dataset_root = Path(data_config.get("path", dataset_yaml.parent))
    for split in ("train", "val"):
        split_path = Path(data_config[split])
        split_path = split_path if split_path.is_absolute() else dataset_root / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"YOLO {split} image path does not exist: {split_path}")

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO training requires the optional deep extra: pip install -e '.[deep]'"
        ) from exc

    if device == "auto":
        selected_device = "0" if torch.cuda.is_available() else "cpu"
    elif device == "0" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU device 0 was requested, but the installed PyTorch build has no CUDA support."
        )
    else:
        selected_device = device

    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat()
    experiment_id = f"{run_name}-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}"
    run_dir = output_dir / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = run_dir / "experiment.json"
    metadata: dict[str, Any] = {
        "experiment_id": experiment_id,
        "status": "RUNNING",
        "started_at": started_at,
        "completed_at": None,
        "config": {
            "dataset_yaml": str(dataset_yaml.resolve()),
            "dataset_yaml_sha256": _sha256(dataset_yaml),
            "manifest_sha256": (
                _sha256(dataset_yaml.parent / "manifest.jsonl")
                if (dataset_yaml.parent / "manifest.jsonl").is_file()
                else None
            ),
            "weights": str(weights.resolve()),
            "weights_sha256": _sha256(weights),
            "epochs": epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "patience": patience,
            "seed": seed,
            "workers": workers,
            "device": selected_device,
            "amp": bool(torch.cuda.is_available()),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "source_sha256": _source_fingerprint(Path(__file__).parents[1]),
        "output_dir": str(run_dir.resolve()),
    }
    _write_json_atomic(metadata_path, metadata)
    model = None
    try:
        model = YOLO(str(weights))
        model_config = getattr(getattr(model, "model", None), "yaml", {})
        metadata["model"] = {
            "wrapper": type(model).__name__,
            "architecture": model_config.get("yaml_file")
            if isinstance(model_config, dict)
            else None,
            "class_names": data_config["names"],
            "initialization_checkpoint": str(weights.resolve()),
            "initialization_checkpoint_sha256": _sha256(weights),
        }
        _write_json_atomic(metadata_path, metadata)
        model.train(
            data=str(dataset_yaml.resolve()),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            patience=patience,
            seed=seed,
            workers=workers,
            device=selected_device,
            amp=bool(torch.cuda.is_available()),
            deterministic=True,
            project=str(run_dir.resolve()),
            name="train",
            exist_ok=False,
            verbose=True,
        )
        trainer = getattr(model, "trainer", None)
        save_dir = Path(getattr(trainer, "save_dir", run_dir / "train"))
        best_checkpoint = save_dir / "weights" / "best.pt"
        last_checkpoint = save_dir / "weights" / "last.pt"
        metadata.update(
            {
                "status": "COMPLETED",
                "completed_at": datetime.now().astimezone().isoformat(),
                "save_dir": str(save_dir.resolve()),
                "best_checkpoint": str(best_checkpoint.resolve())
                if best_checkpoint.is_file()
                else None,
                "last_checkpoint": str(last_checkpoint.resolve())
                if last_checkpoint.is_file()
                else None,
                "metrics_csv": str((save_dir / "results.csv").resolve())
                if (save_dir / "results.csv").is_file()
                else None,
            }
        )
    except Exception as exc:
        metadata.update(
            {
                "status": "FAILED",
                "completed_at": datetime.now().astimezone().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_json_atomic(metadata_path, metadata)
        raise
    finally:
        if model is not None:
            del model
    _write_json_atomic(metadata_path, metadata)
    return metadata_path


def infer_yolo(
    *,
    image: np.ndarray,
    checkpoint: Path,
    tile_size: int = 1024,
    overlap: float = 0.2,
    confidence: float = 0.25,
    nms_iou: float = 0.5,
    image_size: int = 640,
    device: str = "auto",
    class_agnostic_nms: bool = True,
    plate_circle: tuple[float, float, float] | None = None,
    plate_roi_scale: float = 0.92,
    tiled: bool = True,
) -> list[DetectionBox]:
    """Run a local YOLO checkpoint using whole-image or overlapping-tile inference."""

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLO checkpoint does not exist: {checkpoint}")
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D NumPy array")
    if image_size < 32 or tile_size < 1:
        raise ValueError("image_size and tile_size are invalid.")
    if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("confidence and NMS IoU thresholds must be in [0, 1].")
    predictor = load_yolo_predictor(
        checkpoint,
        image_size=image_size,
        confidence=confidence,
        device=device,
        class_agnostic_nms=class_agnostic_nms,
    )
    return infer_yolo_with_predictor(
        image,
        predictor,
        tile_size=tile_size,
        overlap=overlap,
        confidence=confidence,
        nms_iou=nms_iou,
        class_agnostic_nms=class_agnostic_nms,
        plate_circle=plate_circle,
        plate_roi_scale=plate_roi_scale,
        tiled=tiled,
    )


def load_yolo_predictor(
    checkpoint: Path,
    *,
    image_size: int = 640,
    confidence: float = 0.001,
    device: str = "auto",
    class_agnostic_nms: bool = True,
) -> _UltralyticsPredictor:
    """Load one local checkpoint once for efficient manifest-wide prediction."""

    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLO checkpoint does not exist: {checkpoint}")
    if image_size < 32 or not 0 <= confidence <= 1:
        raise ValueError("image_size or confidence is invalid.")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO inference requires the optional deep extra: pip install -e '.[deep]'"
        ) from exc
    if device == "auto":
        selected_device = "0" if torch.cuda.is_available() else "cpu"
    elif device == "0" and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU device 0 was requested, but the installed PyTorch build has no CUDA support."
        )
    else:
        selected_device = device
    try:
        return _UltralyticsPredictor(
            YOLO(str(checkpoint)),
            image_size=image_size,
            confidence=confidence,
            device=selected_device,
            class_agnostic_nms=class_agnostic_nms,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load YOLO checkpoint {checkpoint}: {exc}") from exc


def infer_yolo_with_predictor(
    image: np.ndarray,
    predictor: _UltralyticsPredictor,
    *,
    tile_size: int = 1024,
    overlap: float = 0.2,
    confidence: float = 0.001,
    nms_iou: float = 0.5,
    class_agnostic_nms: bool = True,
    plate_circle: tuple[float, float, float] | None = None,
    plate_roi_scale: float = 0.92,
    tiled: bool = True,
) -> list[DetectionBox]:
    """Infer on one image using an already-loaded predictor and selected geometry."""

    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D NumPy array")
    if tile_size < 1 or not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError("tile size, confidence, or NMS IoU threshold is invalid.")

    if tiled:
        detections = infer_tiled(
            image,
            predictor,
            tile_size=tile_size,
            overlap=overlap,
            confidence_threshold=confidence,
            nms_iou_threshold=nms_iou,
            class_agnostic_nms=class_agnostic_nms,
            plate_circle=plate_circle,
            plate_roi_scale=plate_roi_scale,
        )
    else:
        detections = predictor(image)
        detections = [
            box
            for box in detections
            if 0 <= box.x1 < box.x2 <= image.shape[1] and 0 <= box.y1 < box.y2 <= image.shape[0]
        ]
        detections = filter_plate_exterior(detections, plate_circle, roi_scale=plate_roi_scale)
    return detections


class _UltralyticsPredictor:
    """Callable adapter from Ultralytics results to tile-local DetectionBox objects."""

    def __init__(
        self,
        model: Any,
        *,
        image_size: int,
        confidence: float,
        device: str,
        class_agnostic_nms: bool,
    ) -> None:
        self.model = model
        self.image_size = image_size
        self.confidence = confidence
        self.device = device
        self.class_agnostic_nms = class_agnostic_nms

    def __call__(self, crop: np.ndarray) -> list[DetectionBox]:
        results = self.model.predict(
            source=crop,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            agnostic_nms=self.class_agnostic_nms,
            verbose=False,
            save=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        return [
            DetectionBox(*map(float, xyxy), float(score), int(category))
            for xyxy, score, category in zip(coordinates, scores, classes, strict=True)
        ]


def save_inference_result(
    image: np.ndarray,
    detections: list[DetectionBox],
    *,
    checkpoint: Path,
    output_dir: Path,
    inference_seconds: float,
    tiled: bool,
    confidence: float,
    tile_size: int | None = None,
    overlap: float | None = None,
    nms_iou: float | None = None,
    device: str | None = None,
    class_agnostic_nms: bool = True,
    plate_circle: tuple[float, float, float] | None = None,
) -> tuple[Path, Path]:
    """Save a prediction JSON and overlay with latency and runtime provenance."""

    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    serialized = []
    for detection in detections:
        x1, y1, x2, y2 = map(round, (detection.x1, detection.y1, detection.x2, detection.y2))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 40, 240), 2)
        cv2.putText(
            overlay,
            f"{detection.confidence:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (20, 40, 240),
            1,
        )
        serialized.append(
            {
                "bbox_xyxy": [detection.x1, detection.y1, detection.x2, detection.y2],
                "confidence": detection.confidence,
                "category_id": detection.category_id,
            }
        )
    overlay_path = output_dir / "overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise OSError(f"Could not save inference overlay {overlay_path}")
    result_path = output_dir / "prediction.json"
    payload = {
        "count": len(detections),
        "detections": serialized,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "tiled": tiled,
        "tile_size": tile_size,
        "overlap": overlap,
        "nms_iou": nms_iou,
        "device": device,
        "class_agnostic_nms": class_agnostic_nms,
        "plate_circle": plate_circle,
        "confidence_threshold": confidence,
        "inference_seconds": inference_seconds,
        "hardware": _hardware_report(),
    }
    _write_json_atomic(result_path, payload)
    return result_path, overlay_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_dir.rglob("*.py")):
        digest.update(path.relative_to(source_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _hardware_report() -> dict[str, str | bool | None]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        cuda_available, gpu = False, None
    return {"platform": platform.platform(), "cuda_available": cuda_available, "gpu": gpu}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp_path.replace(path)
