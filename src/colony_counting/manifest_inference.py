"""Run classical or YOLO inference once per image in a frozen manifest split."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from colony_counting.classical import (
    count_colonies,
    detect_plate_circle,
    load_classical_config,
)
from colony_counting.detection.yolo import infer_yolo_with_predictor, load_yolo_predictor
from colony_counting.evaluation import EvalBox, PredictionRecord, write_predictions


def predict_classical_manifest(
    *,
    manifest_path: Path,
    images_root: Path,
    config_path: Path,
    split: str,
    output_path: Path,
) -> dict[str, Any]:
    """Generate prediction JSONL and provenance for exactly one frozen split."""

    rows = load_split_rows(manifest_path, split)
    config = load_classical_config(config_path)
    started = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    records: list[PredictionRecord] = []
    n_total = len(rows)
    for idx, row in enumerate(rows, 1):
        image_path = _image_path(images_root, row)
        t0 = time.perf_counter()
        image = _read_image(image_path)
        result = count_colonies(image, config)
        elapsed = time.perf_counter() - t0
        print(f"[{idx}/{n_total}] {Path(row['image_path']).name}  "
              f"pred={result.count} gt={len(row['boxes'])}  {elapsed:.1f}s", flush=True)
        predictions = tuple(
            EvalBox(float(x), float(y), float(width), float(height))
            for x, y, width, height in (detection.box for detection in result.detections)
        )
        records.append(
            PredictionRecord(
                image_id=str(row["image_id"]),
                gt_count=len(row["boxes"]),
                predicted_count=len(predictions),
                gt_boxes=_ground_truth_boxes(row),
                predicted_boxes=predictions,
                metadata=_record_metadata(row),
            )
        )
    write_predictions(records, output_path)
    return _write_run_metadata(
        manifest_path=manifest_path,
        output_path=output_path,
        rows=rows,
        split=split,
        method="classical",
        config={"config_path": str(config_path), "config": config.__dict__},
        started_seconds=started,
        started_at_utc=started_at_utc,
    )


def predict_yolo_manifest(
    *,
    manifest_path: Path,
    images_root: Path,
    config_path: Path,
    checkpoint: Path,
    split: str,
    output_path: Path,
    confidence_floor: float = 0.001,
    image_size: int = 640,
    tile_size: int = 1024,
    overlap: float = 0.2,
    nms_iou: float = 0.5,
    device: str = "auto",
    class_agnostic_nms: bool = True,
    tiled: bool = True,
) -> dict[str, Any]:
    """Generate score-preserving YOLO predictions while loading the checkpoint only once."""

    rows = load_split_rows(manifest_path, split)
    config = load_classical_config(config_path)
    if not 0 <= confidence_floor <= 1:
        raise ValueError("confidence_floor must be in [0, 1].")
    started = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    predictor = load_yolo_predictor(
        checkpoint,
        image_size=image_size,
        confidence=confidence_floor,
        device=device,
        class_agnostic_nms=class_agnostic_nms,
    )
    class_names = getattr(getattr(predictor, "model", None), "names", {})
    records: list[PredictionRecord] = []
    for row in rows:
        image = _read_image(_image_path(images_root, row))
        plate_circle = detect_plate_circle(image, config)
        detections = infer_yolo_with_predictor(
            image,
            predictor,
            tile_size=tile_size,
            overlap=overlap,
            confidence=confidence_floor,
            nms_iou=nms_iou,
            class_agnostic_nms=class_agnostic_nms,
            plate_circle=plate_circle,
            plate_roi_scale=config.plate_roi_scale,
            tiled=tiled,
        )
        predictions = tuple(
            EvalBox(
                detection.x1,
                detection.y1,
                detection.x2 - detection.x1,
                detection.y2 - detection.y1,
                _category_name(class_names, detection.category_id),
                detection.confidence,
            )
            for detection in detections
        )
        records.append(
            PredictionRecord(
                image_id=str(row["image_id"]),
                gt_count=len(row["boxes"]),
                predicted_count=len(predictions),
                gt_boxes=_ground_truth_boxes(row),
                predicted_boxes=predictions,
                metadata=_record_metadata(row),
            )
        )
    write_predictions(records, output_path)
    return _write_run_metadata(
        manifest_path=manifest_path,
        output_path=output_path,
        rows=rows,
        split=split,
        method="yolo",
        config={
            "checkpoint": str(Path(checkpoint)),
            "confidence_floor": confidence_floor,
            "image_size": image_size,
            "tile_size": tile_size,
            "overlap": overlap,
            "nms_iou": nms_iou,
            "device": device,
            "class_agnostic_nms": class_agnostic_nms,
            "tiled": tiled,
        },
        started_seconds=started,
        started_at_utc=started_at_utc,
        checkpoint=checkpoint,
    )


def load_split_rows(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    """Load rows in one partition and check IDs, groups, box lists, and counts."""

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(manifest_path).read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON at {manifest_path}:{line_number}.") from exc
        if not isinstance(row, dict):
            raise TypeError(f"Manifest line {line_number} must contain an object.")
        if row.get("split") != split:
            continue
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in seen:
            raise ValueError(f"Missing or duplicate image_id in split {split}: {image_id!r}.")
        if not row.get("group_id"):
            raise ValueError(f"Manifest image {image_id} lacks a group_id.")
        boxes = row.get("boxes")
        if not isinstance(boxes, list) or row.get("count") != len(boxes):
            raise ValueError(f"Manifest count/boxes disagree for image {image_id}.")
        seen.add(image_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"No records assigned to manifest split {split!r}.")
    return rows


def _ground_truth_boxes(row: dict[str, Any]) -> tuple[EvalBox, ...]:
    return tuple(
        EvalBox(
            float(box["x"]),
            float(box["y"]),
            float(box["width"]),
            float(box["height"]),
            str(box.get("category_name", box.get("category_id", "unknown"))),
        )
        for box in row["boxes"]
    )


def _record_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_path": row["image_path"],
        "split": row["split"],
        "group_id": str(row["group_id"]),
        "lighting": row.get("lighting"),
        "camera": row.get("camera"),
        "status": row.get("status"),
        "culture_composition": row.get("culture_composition"),
    }


def _category_name(class_names: Any, category_id: int) -> str:
    if isinstance(class_names, dict):
        return str(class_names.get(category_id, class_names.get(str(category_id), category_id)))
    if isinstance(class_names, (list, tuple)) and 0 <= category_id < len(class_names):
        return str(class_names[category_id])
    return str(category_id)


def _image_path(images_root: Path, row: dict[str, Any]) -> Path:
    relative = Path(str(row["image_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Manifest image path must remain inside image root: {relative}.")
    path = Path(images_root) / relative
    if not path.is_file():
        raise FileNotFoundError(f"Manifest image does not exist: {path}")
    return path


def _read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def _write_run_metadata(
    *,
    manifest_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    split: str,
    method: str,
    config: dict[str, Any],
    started_seconds: float,
    started_at_utc: str,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "run_id": f"{method}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "status": "COMPLETED",
        "method": method,
        "split": split,
        "n_images": len(rows),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "prediction_file": str(Path(output_path).resolve()),
        "config": config,
        "source_sha256": _source_fingerprint(Path(__file__).parents[0]),
        "software": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "executable": sys.executable,
        },
        "hardware": _hardware_report(),
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started_seconds,
        "selection_note": "Thresholds and model/config selection must be finalized on validation data before test inference.",
    }
    if checkpoint is not None:
        metadata["checkpoint_sha256"] = _sha256(checkpoint)
    metadata_path = Path(output_path).with_suffix(".run.json")
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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
    except (ImportError, OSError, RuntimeError):
        cuda_available, gpu = False, None
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": cuda_available,
        "gpu": gpu,
    }
