"""Overlapping image tiling and global-coordinate detection merging."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    """A crop window in full-image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class DetectionBox:
    """XYXY box and model score in the coordinate frame supplied by its caller."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    category_id: int = 0

    @property
    def area(self) -> float:
        """Non-negative box area."""

        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def generate_tiles(
    image_width: int, image_height: int, *, tile_size: int, overlap: float = 0.2
) -> list[Tile]:
    """Create a regular overlapping tile grid and anchor the last crop to each far edge."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    tile_width = min(tile_size, image_width)
    tile_height = min(tile_size, image_height)
    stride = max(1, round(tile_size * (1 - overlap)))
    x_starts = _axis_starts(image_width, tile_width, stride)
    y_starts = _axis_starts(image_height, tile_height, stride)
    return [Tile(x, y, tile_width, tile_height) for y in y_starts for x in x_starts]


def to_global_coordinates(
    detection: DetectionBox, tile: Tile, image_width: int, image_height: int
) -> DetectionBox | None:
    """Translate local XYXY coordinates by the tile origin and clip to full image bounds."""

    values = (detection.x1, detection.y1, detection.x2, detection.y2, detection.confidence)
    if not all(math.isfinite(value) for value in values):
        return None
    if detection.x2 <= detection.x1 or detection.y2 <= detection.y1:
        return None
    local_x1 = max(0.0, detection.x1)
    local_y1 = max(0.0, detection.y1)
    local_x2 = min(float(tile.width), detection.x2)
    local_y2 = min(float(tile.height), detection.y2)
    if local_x2 <= local_x1 or local_y2 <= local_y1:
        return None
    x1 = min(float(image_width), tile.x + local_x1)
    y1 = min(float(image_height), tile.y + local_y1)
    x2 = min(float(image_width), tile.x + local_x2)
    y2 = min(float(image_height), tile.y + local_y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return DetectionBox(x1, y1, x2, y2, detection.confidence, detection.category_id)


def merge_tile_detections(
    detections: Iterable[DetectionBox], *, iou_threshold: float = 0.5, class_agnostic: bool = True
) -> list[DetectionBox]:
    """Apply deterministic confidence-ordered NMS across all tile predictions."""

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1].")
    candidates = [
        detection
        for detection in detections
        if detection.area > 0
        and all(
            math.isfinite(value)
            for value in (detection.x1, detection.y1, detection.x2, detection.y2)
        )
        and math.isfinite(detection.confidence)
    ]
    candidates.sort(
        key=lambda item: (-item.confidence, item.x1, item.y1, item.x2, item.y2, item.category_id)
    )
    kept: list[DetectionBox] = []
    for candidate in candidates:
        duplicate = False
        for selected in kept:
            if not class_agnostic and candidate.category_id != selected.category_id:
                continue
            if box_iou(candidate, selected) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def filter_plate_exterior(
    detections: Iterable[DetectionBox],
    plate_circle: tuple[float, float, float] | None,
    *,
    roi_scale: float = 0.92,
) -> list[DetectionBox]:
    """Keep detections whose centers fall inside an optional circular plate interior."""

    if plate_circle is None:
        return list(detections)
    if not 0 < roi_scale <= 1:
        raise ValueError("roi_scale must be in (0, 1].")
    center_x, center_y, radius = plate_circle
    if not all(math.isfinite(value) for value in plate_circle) or radius <= 0:
        raise ValueError("plate circle must contain finite coordinates and a positive radius.")
    radius_squared = (radius * roi_scale) ** 2
    return [
        detection
        for detection in detections
        if (detection.x1 + detection.x2 - 2 * center_x) ** 2 / 4
        + (detection.y1 + detection.y2 - 2 * center_y) ** 2 / 4
        <= radius_squared
    ]


def infer_tiled(
    image: np.ndarray,
    predictor: Callable[[np.ndarray], Iterable[DetectionBox]],
    *,
    tile_size: int,
    overlap: float = 0.2,
    confidence_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    class_agnostic_nms: bool = True,
    plate_circle: tuple[float, float, float] | None = None,
    plate_roi_scale: float = 0.92,
) -> list[DetectionBox]:
    """Run a caller-provided detector on overlapping crops and merge global detections."""

    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3} or image.size == 0:
        raise ValueError("image must be a non-empty 2D or 3D NumPy array")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1].")
    height, width = image.shape[:2]
    candidates: list[DetectionBox] = []
    for tile in generate_tiles(width, height, tile_size=tile_size, overlap=overlap):
        crop = image[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        for local_detection in predictor(crop):
            if local_detection.confidence < confidence_threshold:
                continue
            global_detection = to_global_coordinates(local_detection, tile, width, height)
            if global_detection is not None:
                candidates.append(global_detection)
    merged = merge_tile_detections(
        candidates, iou_threshold=nms_iou_threshold, class_agnostic=class_agnostic_nms
    )
    return filter_plate_exterior(merged, plate_circle, roi_scale=plate_roi_scale)


def box_iou(first: DetectionBox, second: DetectionBox) -> float:
    """Calculate intersection over union for two XYXY boxes."""

    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def _axis_starts(full_size: int, tile_size: int, stride: int) -> list[int]:
    if tile_size >= full_size:
        return [0]
    starts = list(range(0, full_size - tile_size + 1, stride))
    last = full_size - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts
