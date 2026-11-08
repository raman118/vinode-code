"""Run a one-epoch YOLO integration smoke on generated toy images, never AGAR data."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from colony_counting.detection.yolo import infer_yolo, save_inference_result, train_yolo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results")
        / f"synthetic-yolo-smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    args = parser.parse_args()
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke artifacts: {output_dir}")
    dataset_dir = output_dir / "dataset"
    for split, count in (("train", 2), ("val", 1)):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image_index in range(count):
            image = np.full((96, 96, 3), 225, dtype=np.uint8)
            center = (48 + image_index, 48)
            cv2.circle(image, center, 6, (30, 30, 30), -1)
            image_path = dataset_dir / "images" / split / f"toy-{image_index}.png"
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"Could not write synthetic image {image_path}")
            label_path = dataset_dir / "labels" / split / f"toy-{image_index}.txt"
            center_x = center[0] / 96
            label_path.write_text(f"0 {center_x:.8f} 0.5 0.125 0.125\n", encoding="utf-8")

    dataset_yaml = dataset_dir / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "toy-object"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    initial_weights = output_dir / "random-init.pt"
    initial_model = YOLO("yolo11n.yaml")
    initial_model.save(str(initial_weights))
    del initial_model

    metadata_path = train_yolo(
        dataset_yaml=dataset_yaml,
        weights=initial_weights,
        output_dir=output_dir / "training",
        epochs=1,
        image_size=64,
        batch_size=2,
        patience=1,
        seed=42,
        workers=0,
        device="0",
        run_name="synthetic-smoke",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint = Path(metadata["best_checkpoint"] or "")
    if not checkpoint.is_file():
        raise RuntimeError("Synthetic smoke training completed without a saved best checkpoint.")

    inference_image = cv2.imread(str(dataset_dir / "images" / "val" / "toy-0.png"))
    inference_started = time.perf_counter()
    detections = infer_yolo(
        image=inference_image,
        checkpoint=checkpoint,
        tile_size=96,
        overlap=0,
        confidence=0.01,
        image_size=64,
        device="0",
        tiled=True,
    )
    inference_seconds = time.perf_counter() - inference_started
    prediction, overlay = save_inference_result(
        inference_image,
        detections,
        checkpoint=checkpoint,
        output_dir=output_dir / "inference",
        inference_seconds=inference_seconds,
        tiled=True,
        confidence=0.01,
        tile_size=96,
        overlap=0,
        nms_iou=0.5,
        device="0",
    )
    report = {
        "status": "SMOKE_VERIFIED",
        "data_kind": "generated synthetic toy images",
        "is_agar_evidence": False,
        "trained_epochs": 1,
        "initialization": "random YOLO11n architecture from package YAML; no pretrained checkpoint",
        "initial_weights": str(initial_weights),
        "training_metadata": str(metadata_path),
        "checkpoint": str(checkpoint),
        "inference_prediction": str(prediction),
        "overlay": str(overlay),
        "predicted_count": len(detections),
        "scientific_performance_claim": None,
    }
    report_path = output_dir / "smoke-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
