"""Synthetic geometry checks for tiled detector post-processing, not model results."""

import numpy as np

from colony_counting.detection.tiling import (
    DetectionBox,
    Tile,
    filter_plate_exterior,
    generate_tiles,
    infer_tiled,
    merge_tile_detections,
    to_global_coordinates,
)


def test_tiles_cover_far_edges_and_small_images():
    tiles = generate_tiles(1000, 700, tile_size=400, overlap=0.25)
    assert max(tile.x + tile.width for tile in tiles) == 1000
    assert max(tile.y + tile.height for tile in tiles) == 700
    assert len(tiles) == 6
    assert generate_tiles(300, 200, tile_size=400) == [Tile(0, 0, 300, 200)]


def test_local_boxes_translate_and_clip_to_full_image():
    tile = Tile(300, 100, 400, 400)
    local = DetectionBox(120, 150, 140, 170, 0.9)
    assert to_global_coordinates(local, tile, 1000, 700) == DetectionBox(420, 250, 440, 270, 0.9)
    clipped = to_global_coordinates(DetectionBox(390, 390, 420, 420, 0.5), tile, 650, 450)
    assert clipped is None


def test_nms_merges_overlapping_tile_duplicates():
    detections = [
        DetectionBox(20, 20, 40, 40, 0.8),
        DetectionBox(21, 21, 41, 41, 0.95),
        DetectionBox(70, 70, 80, 80, 0.6),
    ]
    merged = merge_tile_detections(detections, iou_threshold=0.5)
    assert len(merged) == 2
    assert merged[0].confidence == 0.95


def test_plate_exterior_filter_uses_detection_center():
    detections = [
        DetectionBox(45, 45, 55, 55, 0.8),
        DetectionBox(90, 90, 100, 100, 0.9),
    ]
    kept = filter_plate_exterior(detections, (50, 50, 50), roi_scale=0.9)
    assert len(kept) == 1
    assert kept[0].x1 == 45


def test_infer_tiled_maps_and_deduplicates_real_predictor_outputs():
    image = np.zeros((100, 200), dtype=np.uint8)
    image[45:56, 85:96] = 255

    def predictor(crop):
        ys, xs = np.where(crop > 0)
        if len(xs) == 0:
            return []
        return [
            DetectionBox(
                float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1), 0.9
            )
        ]

    detections = infer_tiled(
        image,
        predictor,
        tile_size=120,
        overlap=0.5,
        confidence_threshold=0.1,
        nms_iou_threshold=0.5,
    )
    assert len(detections) == 1
    assert detections[0].x1 == 85
    assert detections[0].y1 == 45
