"""Configurable classical CV colony counter with inspectable image-processing stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class ClassicalConfig:
    """Parameters for classical colony segmentation.

    Pixel-scale settings depend on source resolution and must be selected on training/validation
    data. Defaults are conservative starting values, not tuned AGAR parameters.
    """

    channel: str = "gray"
    illumination_correction: bool = True
    background_sigma: float = 31.0
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    threshold_method: str = "otsu"
    polarity: str = "dark"
    adaptive_block_size: int = 51
    adaptive_constant: float = 3.0
    morphology_kernel: int = 3
    opening_iterations: int = 1
    closing_iterations: int = 1
    plate_detection: bool = True
    plate_radius_min_fraction: float = 0.32
    plate_radius_max_fraction: float = 0.51
    plate_roi_scale: float = 0.92
    plate_hough_param2: float = 34.0
    peak_min_distance: int = 7
    peak_min_fraction: float = 0.18
    min_colony_area: int = 8
    max_colony_area: int = 10000
    cluster_area_correction: bool = False
    nominal_colony_area: float = 300.0
    cluster_correction_ratio: float = 1.8

    def __post_init__(self) -> None:
        """Reject invalid settings early instead of silently changing their meaning."""

        if self.channel not in {"gray", "blue", "green", "red", "h", "s", "v", "lab_l"}:
            raise ValueError(f"Unsupported channel {self.channel!r}.")
        if self.threshold_method not in {"otsu", "adaptive"}:
            raise ValueError("threshold_method must be 'otsu' or 'adaptive'.")
        if self.polarity not in {"dark", "bright"}:
            raise ValueError("polarity must be 'dark' or 'bright'.")
        if self.morphology_kernel < 1 or self.morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be a positive odd integer.")
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be an odd integer of at least 3.")
        if self.background_sigma < 0:
            raise ValueError("background_sigma cannot be negative.")
        if self.clahe_clip_limit <= 0 or self.clahe_grid_size < 1:
            raise ValueError("CLAHE clip limit and grid size must be positive.")
        if self.opening_iterations < 0 or self.closing_iterations < 0:
            raise ValueError("Morphology iteration counts cannot be negative.")
        if self.peak_min_distance < 1:
            raise ValueError("peak_min_distance must be positive.")
        if self.min_colony_area < 1 or self.max_colony_area < self.min_colony_area:
            raise ValueError("Colony area limits are invalid.")
        if not 0 < self.plate_roi_scale <= 1:
            raise ValueError("plate_roi_scale must be in (0, 1].")
        if not 0 < self.plate_radius_min_fraction <= self.plate_radius_max_fraction <= 1:
            raise ValueError("Plate radius fractions must be ordered and in (0, 1].")
        if self.plate_hough_param2 <= 0:
            raise ValueError("plate_hough_param2 must be positive.")
        if not 0 < self.peak_min_fraction <= 1:
            raise ValueError("peak_min_fraction must be in (0, 1].")
        if self.nominal_colony_area <= 0 or self.cluster_correction_ratio <= 1:
            raise ValueError("Cluster correction area parameters are invalid.")


@dataclass(frozen=True)
class Detection:
    """One detected colony represented by center and absolute XYWH box."""

    x: float
    y: float
    box: tuple[int, int, int, int]
    area: float
    source: str = "watershed"


@dataclass
class ClassicalResult:
    """Classical method output and intermediate diagnostic images."""

    detections: list[Detection]
    plate_circle: tuple[int, int, int] | None
    stages: dict[str, np.ndarray]

    @property
    def count(self) -> int:
        """Number of colony detections."""

        return len(self.detections)


def load_classical_config(path: Path) -> ClassicalConfig:
    """Load validated classical parameters from YAML."""

    try:
        payload: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot load classical config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Classical YAML config must contain a mapping.")
    unknown = set(payload) - set(ClassicalConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown classical config keys: {sorted(unknown)}")
    return ClassicalConfig(**payload)


def detect_plate_circle(image: np.ndarray, config: ClassicalConfig) -> tuple[int, int, int] | None:
    """Estimate the plate circle using Hough circles, or return ``None`` on failure."""

    if not config.plate_detection:
        return None
    gray = _to_gray(image)
    min_dim = min(gray.shape[:2])
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim * 0.5,
        param1=100,
        param2=config.plate_hough_param2,
        minRadius=max(1, int(min_dim * config.plate_radius_min_fraction)),
        maxRadius=max(2, int(min_dim * config.plate_radius_max_fraction)),
    )
    if circles is None or circles.size == 0:
        return None
    candidates = np.rint(circles[0]).astype(int)
    # Prefer the largest plausible circle; plate rim is generally the outer boundary.
    x, y, radius = max(candidates, key=lambda circle: circle[2])
    return int(x), int(y), int(radius)


def circular_roi_mask(
    shape: tuple[int, int], circle: tuple[int, int, int] | None, roi_scale: float
) -> np.ndarray:
    """Build a circular interior mask; fall back to the inscribed image circle."""

    height, width = shape
    if circle is None:
        cx, cy = width // 2, height // 2
        radius = min(width, height) * 0.5
    else:
        cx, cy, radius = circle
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), max(1, int(radius * roi_scale)), 255, -1)
    return mask


def select_channel(image: np.ndarray, channel: str) -> np.ndarray:
    """Select a reproducible grayscale or color-space channel from a BGR image."""

    if image.ndim == 2:
        if channel in {"gray", "lab_l"}:
            return image.copy()
        raise ValueError(f"Channel {channel!r} requires a three-channel BGR image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected grayscale or BGR image, received shape {image.shape}.")
    if channel == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channel in {"blue", "green", "red"}:
        return image[:, :, {"blue": 0, "green": 1, "red": 2}[channel]].copy()
    if channel in {"h", "s", "v"}:
        hsv_index = {"h": 0, "s": 1, "v": 2}[channel]
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, hsv_index]
    if channel == "lab_l":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    raise ValueError(f"Unsupported channel {channel!r}.")


def correct_illumination(channel: np.ndarray, roi_mask: np.ndarray, sigma: float) -> np.ndarray:
    """Flatten low-frequency illumination by subtracting a Gaussian background estimate."""

    if sigma <= 0:
        return channel.copy()
    source = channel.astype(np.float32)
    background = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
    roi_values = source[roi_mask > 0]
    reference = float(np.median(roi_values)) if roi_values.size else float(np.median(source))
    corrected = np.clip(source - background + reference, 0, 255).astype(np.uint8)
    corrected[roi_mask == 0] = 0
    return corrected


def normalize_contrast(
    channel: np.ndarray, roi_mask: np.ndarray, clip_limit: float, grid_size: int
) -> np.ndarray:
    """Apply CLAHE then normalize the ROI intensity range to uint8."""

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(channel)
    values = enhanced[roi_mask > 0]
    if values.size == 0:
        raise ValueError("ROI mask contains no pixels.")
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        output = np.zeros_like(enhanced)
    else:
        output = np.clip(
            (enhanced.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255
        ).astype(np.uint8)
    output[roi_mask == 0] = 0
    return output


def threshold_channel(
    channel: np.ndarray, roi_mask: np.ndarray, config: ClassicalConfig
) -> np.ndarray:
    """Threshold a normalized channel using Otsu or adaptive mean thresholding."""

    threshold_type = cv2.THRESH_BINARY_INV if config.polarity == "dark" else cv2.THRESH_BINARY
    if config.threshold_method == "otsu":
        _, binary = cv2.threshold(channel, 0, 255, threshold_type | cv2.THRESH_OTSU)
    elif config.threshold_method == "adaptive":
        block_size = min(config.adaptive_block_size, min(channel.shape[:2]) | 1)
        block_size = max(3, block_size)
        binary = cv2.adaptiveThreshold(
            channel,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            threshold_type,
            block_size,
            config.adaptive_constant,
        )
    else:  # Defensive if callers bypass dataclass validation.
        raise ValueError(f"Unsupported threshold method: {config.threshold_method}")
    binary[roi_mask == 0] = 0
    return binary


def clean_mask(
    binary: np.ndarray, kernel_size: int, opening_iterations: int, closing_iterations: int
) -> np.ndarray:
    """Remove speckles and close small gaps with elliptical morphology."""

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)


def distance_transform(binary: np.ndarray) -> np.ndarray:
    """Compute precise Euclidean distance to the foreground boundary."""

    return cv2.distanceTransform(binary, cv2.DIST_L2, 5)


def extract_markers(
    distance: np.ndarray, binary: np.ndarray, min_distance: int, min_fraction: float
) -> np.ndarray:
    """Find local maxima of the distance transform and label them as watershed seeds."""

    size = 2 * min_distance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    local_max = cv2.dilate(distance, kernel)
    max_value = float(distance.max())
    if max_value <= 0:
        return np.zeros(binary.shape, dtype=np.int32)
    peak_floor = max(1.0, max_value * min_fraction)
    peaks = (distance >= local_max - 1e-6) & (distance >= peak_floor) & (binary > 0)
    _, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    return markers.astype(np.int32)


def watershed_labels(image: np.ndarray, binary: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Separate touching foreground regions using OpenCV watershed."""

    if markers.max(initial=0) == 0:
        return np.zeros(binary.shape, dtype=np.int32)
    # Every foreground pixel must be unknown so watershed can grow each colony seed across
    # its component. Marking the entire binary interior as label 1 incorrectly fixes it as
    # background and leaves only the seed pixels to be measured.
    seeded = np.zeros(binary.shape, dtype=np.int32)
    seeded[binary == 0] = 1
    seeded[markers > 0] = markers[markers > 0] + 1
    color = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color.copy(), seeded)
    seeded[(binary == 0) | (seeded < 2)] = 0
    return seeded


def count_colonies(image: np.ndarray, config: ClassicalConfig | None = None) -> ClassicalResult:
    """Run the complete configurable classical segmentation and counting pipeline."""

    config = config or ClassicalConfig()
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image must be a non-empty NumPy array.")
    if image.dtype != np.uint8:
        raise ValueError(f"Input image must use uint8 pixels; received {image.dtype}.")
    if image.ndim not in {2, 3} or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError(f"Expected a grayscale or BGR image; received shape {image.shape}.")

    circle = detect_plate_circle(image, config)
    roi = circular_roi_mask(image.shape[:2], circle, config.plate_roi_scale)
    channel = select_channel(image, config.channel)
    corrected = (
        correct_illumination(channel, roi, config.background_sigma)
        if config.illumination_correction
        else channel.copy()
    )
    corrected[roi == 0] = 0
    normalized = normalize_contrast(corrected, roi, config.clahe_clip_limit, config.clahe_grid_size)
    binary = threshold_channel(normalized, roi, config)
    cleaned = clean_mask(
        binary, config.morphology_kernel, config.opening_iterations, config.closing_iterations
    )
    cleaned[roi == 0] = 0
    distance = distance_transform(cleaned)
    markers = extract_markers(distance, cleaned, config.peak_min_distance, config.peak_min_fraction)
    labels = watershed_labels(image, cleaned, markers)
    detections = _detections_from_labels(labels, markers, cleaned, distance, config)
    if config.cluster_area_correction:
        detections = _correct_large_clusters(detections, cleaned, distance, config)

    display_distance = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stage_images = {
        "roi": roi,
        "channel": channel,
        "illumination_corrected": corrected,
        "contrast_normalized": normalized,
        "threshold": binary,
        "morphology": cleaned,
        "distance_transform": display_distance,
        "markers": _colorize_labels(markers),
        "watershed": _colorize_labels(labels),
        "overlay": draw_detections(image, detections, circle),
    }
    return ClassicalResult(detections=detections, plate_circle=circle, stages=stage_images)


def save_diagnostics(result: ClassicalResult, output_dir: Path) -> list[Path]:
    """Write every diagnostic stage to disk and return the saved paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, image in result.stages.items():
        path = output_dir / f"{name}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write diagnostic image {path}.")
        outputs.append(path)
    return outputs


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    plate_circle: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Draw estimated colony boxes and centers over a copy of the input image."""

    output = image.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    for detection in detections:
        x, y, width, height = detection.box
        cv2.rectangle(output, (x, y), (x + width, y + height), (20, 40, 240), 1)
        cv2.circle(output, (round(detection.x), round(detection.y)), 2, (0, 255, 0), -1)
    if plate_circle is not None:
        cv2.circle(output, plate_circle[:2], plate_circle[2], (255, 180, 0), 1)
    return output


def _detections_from_labels(
    labels: np.ndarray,
    markers: np.ndarray,
    binary: np.ndarray,
    distance: np.ndarray,
    config: ClassicalConfig,
) -> list[Detection]:
    detections: list[Detection] = []
    region_ids = np.unique(labels)
    for region_id in region_ids:
        if region_id < 2:
            continue
        region = (labels == region_id).astype(np.uint8)
        area = float(cv2.countNonZero(region))
        if area < config.min_colony_area or area > config.max_colony_area:
            continue
        ys, xs = np.where(region > 0)
        if not len(xs):
            continue
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        detections.append(
            Detection(
                center_x, center_y, (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1), area
            )
        )
    return detections


def _correct_large_clusters(
    detections: list[Detection], binary: np.ndarray, distance: np.ndarray, config: ClassicalConfig
) -> list[Detection]:
    """Add distance-peak detections when a connected cluster's area implies missed colonies."""

    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = list(detections)
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < config.nominal_colony_area * config.cluster_correction_ratio:
            continue
        region = component_labels == component_id
        current = sum(region[round(detection.y), round(detection.x)] for detection in output)
        expected = max(1, round(area / config.nominal_colony_area))
        missing = expected - current
        if missing <= 0:
            continue
        peak_map = np.where(region, distance, 0).astype(np.float32)
        radius = max(1, round(np.sqrt(config.nominal_colony_area / np.pi) * 0.7))
        for _ in range(missing):
            _, peak_value, _, location = cv2.minMaxLoc(peak_map)
            if peak_value <= 0:
                break
            cx, cy = location
            half = max(1, round(np.sqrt(config.nominal_colony_area) / 2))
            x0, y0 = max(0, cx - half), max(0, cy - half)
            width = min(binary.shape[1] - x0, half * 2 + 1)
            height = min(binary.shape[0] - y0, half * 2 + 1)
            output.append(
                Detection(
                    float(cx),
                    float(cy),
                    (x0, y0, width, height),
                    config.nominal_colony_area,
                    "area-correction",
                )
            )
            cv2.circle(peak_map, (cx, cy), radius, 0, -1)
    return output


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Create deterministic pseudo-color labels for visual diagnostics."""

    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels):
        if label <= 0:
            continue
        color = ((label * 67) % 255, (label * 131) % 255, (label * 197) % 255)
        output[labels == label] = color
    return output


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
