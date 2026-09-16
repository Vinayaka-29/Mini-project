from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_LOCATIONS = [
    Path(__file__).resolve().parents[3] / "configs" / "parking_layout.json",
    Path(__file__).resolve().parents[2] / "configs" / "parking_layout.json",
    Path.cwd() / "configs" / "parking_layout.json",
    Path.cwd() / "Mini-project" / "configs" / "parking_layout.json",
]


def load_parking_layout(config_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load parking spot definitions and polygon coordinates from parking_layout.json."""
    candidate_paths: list[Path] = []
    if config_path:
        candidate_paths.append(Path(config_path))
    candidate_paths.extend(DEFAULT_CONFIG_LOCATIONS)

    for path in candidate_paths:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                slots = data.get("slots", [])
                logger.info("Successfully loaded %d slots from %s", len(slots), path)
                return slots
            except Exception as exc:
                logger.error("Failed to parse parking layout from %s: %s", path, exc)

    logger.warning("Could not locate parking_layout.json configuration file.")
    return []


def is_point_in_polygon(point: tuple[float, float] | list[float], polygon: list[list[float]]) -> bool:
    """Check if a 2D point (x, y) is inside or on the boundary of a polygon."""
    if len(polygon) < 3:
        return False
    poly_np = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(poly_np, (float(point[0]), float(point[1])), False) >= 0


def calculate_box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Calculate standard Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - intersection

    return intersection / max(1e-6, union)


def calculate_polygon_bbox_overlap(
    polygon: list[list[float]],
    bbox: list[float],
) -> tuple[float, float, bool]:
    """
    Calculate intersection area using localized pixel masks, overlap ratio (intersection / slot_area),
    IoU, and whether vehicle center falls inside the slot polygon.
    """
    if len(polygon) < 3 or len(bbox) < 4:
        return 0.0, 0.0, False

    vx1, vy1, vx2, vy2 = bbox

    # Determine localized bounding dimensions to avoid creating a massive full-image mask
    poly_xs = [float(p[0]) for p in polygon]
    poly_ys = [float(p[1]) for p in polygon]

    min_x = int(max(0, min(min(poly_xs), vx1)))
    min_y = int(max(0, min(min(poly_ys), vy1)))
    max_x = int(max(max(poly_xs), vx2)) + 2
    max_y = int(max(max(poly_ys), vy2)) + 2

    local_w = max(1, max_x - min_x)
    local_h = max(1, max_y - min_y)

    # 1. Create two localized blank np.uint8 masks
    mask_poly = np.zeros((local_h, local_w), dtype=np.uint8)
    mask_bbox = np.zeros((local_h, local_w), dtype=np.uint8)

    # 2. Draw parking spot polygon on first mask (value 255)
    local_poly = np.array(
        [[int(round(px - min_x)), int(round(py - min_y))] for px, py in zip(poly_xs, poly_ys)],
        dtype=np.int32,
    )
    cv2.fillPoly(mask_poly, [local_poly], 255)

    # 3. Draw vehicle bounding box on second mask (value 255)
    local_vx1 = int(round(max(0, vx1 - min_x)))
    local_vy1 = int(round(max(0, vy1 - min_y)))
    local_vx2 = int(round(max(0, vx2 - min_x)))
    local_vy2 = int(round(max(0, vy2 - min_y)))
    cv2.rectangle(mask_bbox, (local_vx1, local_vy1), (local_vx2, local_vy2), 255, -1)

    # 4. Use bitwise_and to find exact overlapping pixels
    intersection_mask = cv2.bitwise_and(mask_poly, mask_bbox)

    # 5. Count non-zero pixels for intersection area
    intersection_area = float(cv2.countNonZero(intersection_mask))
    slot_area = float(cv2.countNonZero(mask_poly))
    if slot_area <= 0:
        slot_area = max(1.0, (max(poly_xs) - min(poly_xs)) * (max(poly_ys) - min(poly_ys)))

    vehicle_area = float(max(1.0, (vx2 - vx1) * (vy2 - vy1)))

    overlap_ratio = intersection_area / max(1.0, slot_area)
    union_area = max(1.0, slot_area + vehicle_area - intersection_area)
    iou = intersection_area / union_area

    # Center point in polygon check
    center_x = (vx1 + vx2) / 2.0
    center_y = (vy1 + vy2) / 2.0
    slot_np = np.array(polygon, dtype=np.float32)
    center_in_poly = cv2.pointPolygonTest(slot_np, (center_x, center_y), False) >= 0

    return float(overlap_ratio), float(iou), bool(center_in_poly)


class SlotOccupancyEngine:
    """Calculates parking slot occupancy by matching YOLO vehicle detections with predefined slot polygons."""

    def __init__(self, config_path: str | Path | None = None, iou_threshold: float = 0.50) -> None:
        self.iou_threshold = iou_threshold
        self.default_slots: list[dict[str, Any]] = load_parking_layout(config_path)

    def set_layout(self, slots: list[dict[str, Any]] | dict[str, Any]) -> None:
        """Dynamically update the default slot definitions in memory."""
        if isinstance(slots, dict):
            self.default_slots = slots.get("slots", [])
        else:
            self.default_slots = list(slots)


    def compute_slot_status(
        self,
        slot: dict[str, Any],
        detections: list[Any],
        image_shape: tuple[int, int] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Determine if a single parking slot is OCCUPIED or AVAILABLE based on vehicle bounding boxes.
        Applies IoU and Point-in-Polygon overlap logic.
        """
        active_threshold = threshold if threshold is not None else self.iou_threshold
        raw_poly = slot.get("polygon", [])
        if not raw_poly or len(raw_poly) < 3:
            return {
                "slot_id": slot.get("slot_id", "UNKNOWN"),
                "status": slot.get("status", "AVAILABLE"),
                "confidence": 0.95,
                "occupancy_score": 0.0,
                "iou": 0.0,
            }

        poly: list[list[float]] = []
        if image_shape:
            h, w = image_shape
            for pt in raw_poly:
                px = pt[0] * w if 0 <= pt[0] <= 1.0 else pt[0]
                py = pt[1] * h if 0 <= pt[1] <= 1.0 else pt[1]
                poly.append([float(px), float(py)])
        else:
            poly = [[float(pt[0]), float(pt[1])] for pt in raw_poly]

        best_overlap = 0.0
        best_iou = 0.0
        best_conf = 0.0
        matched_det = None

        for det in detections:
            bbox = getattr(det, "bbox", None) or (det.get("bbox") if isinstance(det, dict) else None)
            if not bbox or len(bbox) < 4:
                continue

            conf = getattr(det, "confidence", 0.0) if hasattr(det, "confidence") else det.get("confidence", 0.0)
            overlap_ratio, iou, center_inside = calculate_polygon_bbox_overlap(poly, bbox)

            score = overlap_ratio
            if center_inside:
                score = max(score, 0.55)

            if score > best_overlap:
                best_overlap = score
                best_iou = iou
                best_conf = conf
                matched_det = det

        is_occupied = (best_overlap >= active_threshold) or (best_iou >= 0.35) or (best_overlap >= 0.20 and best_conf > 0.40)
        status = "OCCUPIED" if is_occupied else "AVAILABLE"
        confidence = round(max(0.75, min(0.99, best_conf if is_occupied else (1.0 - min(1.0, best_overlap)))), 2)

        return {
            "slot_id": slot["slot_id"],
            "status": status,
            "confidence": confidence,
            "occupancy_score": round(best_overlap, 3),
            "iou": round(best_iou, 3),
            "matched_vehicle": matched_det.to_dict() if hasattr(matched_det, "to_dict") else (matched_det if isinstance(matched_det, dict) else None),
        }

    def map_detections_to_spots(
        self,
        slots: list[dict[str, Any]],
        detections: list[Any],
        image_shape: tuple[int, int] | None = None,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Map all detected vehicles to all parking spots and return updated slot states."""
        target_slots = slots if slots else self.default_slots
        results: list[dict[str, Any]] = []

        for slot in target_slots:
            evaluation = self.compute_slot_status(slot, detections, image_shape=image_shape, threshold=threshold)
            updated = dict(slot)
            updated["status"] = evaluation["status"]
            updated["confidence"] = evaluation["confidence"]
            updated["occupancy_score"] = evaluation["occupancy_score"]
            results.append(updated)

        return results


# Global instance for shared occupancy evaluation
global_occupancy_engine = SlotOccupancyEngine()
