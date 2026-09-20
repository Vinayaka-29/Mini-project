from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.vision.occupancy import scale_polygon, _is_normalized

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_PATH = Path(__file__).resolve().parents[3] / "configs" / "reference_frame.npy"

# Thresholds for occupancy decision
MEAN_DIFF_THRESHOLD = 25.0  # Mean pixel intensity difference in ROI
EDGE_RATIO_THRESHOLD = 1.6  # Edge pixel density ratio (current / reference)
MIN_PIXEL_COUNT = 50  # Minimum non-zero pixels in ROI mask to consider valid


class ReferenceFrameDetector:
    """Fixed-camera background-subtraction detector.

    Compares each parking bay's ROI against an empty-lot reference frame
    using absolute difference and Canny edge density.  This is far more
    reliable than generic COCO YOLO on a miniature/cardboard parking lot.
    """

    def __init__(self, reference_path: str | Path | None = None) -> None:
        self.reference_gray: np.ndarray | None = None
        self.reference_edges: np.ndarray | None = None
        self._reference_path = Path(reference_path) if reference_path else DEFAULT_REFERENCE_PATH
        self._try_load_reference()

    # ------------------------------------------------------------------
    # Reference frame management
    # ------------------------------------------------------------------

    def _try_load_reference(self) -> None:
        """Attempt to load a persisted reference frame from disk."""
        if self._reference_path.exists():
            try:
                ref = np.load(str(self._reference_path))
                self._set_internal(ref)
                logger.info("Loaded reference frame from %s (%dx%d)", self._reference_path, ref.shape[1], ref.shape[0])
            except Exception as exc:
                logger.warning("Could not load reference frame from %s: %s", self._reference_path, exc)

    def set_reference(self, frame: np.ndarray) -> None:
        """Store a new empty-lot reference frame and persist to disk."""
        if frame is None or frame.size == 0:
            logger.error("Cannot set reference: empty frame provided.")
            return
        self._set_internal(frame)
        self.save_reference()
        logger.info("Reference frame set and saved (%dx%d)", frame.shape[1], frame.shape[0])

    def _set_internal(self, frame: np.ndarray) -> None:
        """Convert frame to grayscale and pre-compute edges."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        self.reference_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        self.reference_edges = cv2.Canny(self.reference_gray, 50, 150)

    def save_reference(self, path: str | Path | None = None) -> None:
        """Persist the reference frame to a .npy file."""
        target = Path(path) if path else self._reference_path
        if self.reference_gray is None:
            logger.warning("No reference frame to save.")
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(target), self.reference_gray)
            logger.info("Reference frame persisted to %s", target)
        except Exception as exc:
            logger.error("Failed to save reference frame to %s: %s", target, exc)

    def load_reference(self, path: str | Path | None = None) -> bool:
        """Load reference frame from disk. Returns True on success."""
        target = Path(path) if path else self._reference_path
        if not target.exists():
            logger.warning("Reference frame file not found: %s", target)
            return False
        try:
            ref = np.load(str(target))
            self._set_internal(ref)
            logger.info("Reference frame loaded from %s", target)
            return True
        except Exception as exc:
            logger.error("Failed to load reference frame from %s: %s", target, exc)
            return False

    @property
    def has_reference(self) -> bool:
        return self.reference_gray is not None

    # ------------------------------------------------------------------
    # Per-bay occupancy detection
    # ------------------------------------------------------------------

    def _extract_roi_mask(
        self,
        frame_gray: np.ndarray,
        polygon: list[list[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a polygon mask and extract the masked ROI region."""
        h, w = frame_gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in polygon], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask, cv2.bitwise_and(frame_gray, mask)

    def detect_bay_occupancy(
        self,
        frame: np.ndarray,
        slot: dict[str, Any],
        image_shape: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Determine if a single bay is occupied by comparing against the reference frame.

        Returns a result dict with status, confidence, mean_diff, edge_ratio.
        """
        slot_id = slot.get("slot_id", "UNKNOWN")
        raw_poly = slot.get("polygon", [])

        if not self.has_reference:
            logger.warning("BAY %s: No reference frame — returning UNKNOWN", slot_id)
            return {
                "slot_id": slot_id,
                "status": "UNKNOWN",
                "confidence": 0.0,
                "mean_diff": 0.0,
                "edge_ratio": 0.0,
            }

        if not raw_poly or len(raw_poly) < 3:
            return {
                "slot_id": slot_id,
                "status": "AVAILABLE",
                "confidence": 0.95,
                "mean_diff": 0.0,
                "edge_ratio": 0.0,
            }

        # Determine image shape for scaling
        if image_shape is None:
            image_shape = frame.shape[:2]

        # Scale normalized polygon to absolute coords
        poly = scale_polygon(raw_poly, image_shape)

        # Convert current frame to grayscale
        if len(frame.shape) == 3:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = frame.copy()
        current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

        # Resize reference to match current frame if needed
        ref_gray = self.reference_gray
        if ref_gray.shape != current_gray.shape:
            ref_gray = cv2.resize(ref_gray, (current_gray.shape[1], current_gray.shape[0]))

        # Extract ROI masks
        mask, current_roi = self._extract_roi_mask(current_gray, poly)
        _, ref_roi = self._extract_roi_mask(ref_gray, poly)

        # Count valid pixels in the ROI
        valid_pixels = cv2.countNonZero(mask)
        if valid_pixels < MIN_PIXEL_COUNT:
            logger.debug("BAY %s: ROI too small (%d px), treating as AVAILABLE", slot_id, valid_pixels)
            return {
                "slot_id": slot_id,
                "status": "AVAILABLE",
                "confidence": 0.90,
                "mean_diff": 0.0,
                "edge_ratio": 0.0,
            }

        # 1. Absolute difference in the ROI
        diff = cv2.absdiff(current_roi, ref_roi)
        # Only consider pixels inside the polygon mask
        diff_masked = cv2.bitwise_and(diff, mask)
        mean_diff = float(np.sum(diff_masked)) / max(1, valid_pixels)

        # 2. Edge density comparison
        current_edges = cv2.Canny(current_gray, 50, 150)
        current_edge_roi = cv2.bitwise_and(current_edges, mask)
        ref_edges = cv2.Canny(ref_gray, 50, 150)
        ref_edge_roi = cv2.bitwise_and(ref_edges, mask)

        current_edge_count = float(cv2.countNonZero(current_edge_roi))
        ref_edge_count = float(cv2.countNonZero(ref_edge_roi))
        edge_ratio = current_edge_count / max(1.0, ref_edge_count)

        # Decision
        is_occupied = (mean_diff > MEAN_DIFF_THRESHOLD) or (edge_ratio > EDGE_RATIO_THRESHOLD)
        status = "OCCUPIED" if is_occupied else "AVAILABLE"

        # Confidence based on how far above thresholds we are
        if is_occupied:
            diff_factor = min(1.0, mean_diff / (MEAN_DIFF_THRESHOLD * 3))
            edge_factor = min(1.0, edge_ratio / (EDGE_RATIO_THRESHOLD * 2))
            confidence = round(0.75 + 0.24 * max(diff_factor, edge_factor), 2)
        else:
            confidence = round(0.85 + 0.14 * (1.0 - min(1.0, mean_diff / MEAN_DIFF_THRESHOLD)), 2)

        # Auditable logging
        logger.info(
            "BAY %s: status=%s mean_diff=%.1f edge_ratio=%.2f "
            "threshold_diff=%.1f threshold_edge=%.1f confidence=%.2f",
            slot_id, status, mean_diff, edge_ratio,
            MEAN_DIFF_THRESHOLD, EDGE_RATIO_THRESHOLD, confidence,
        )

        return {
            "slot_id": slot_id,
            "status": status,
            "confidence": confidence,
            "mean_diff": round(mean_diff, 2),
            "edge_ratio": round(edge_ratio, 2),
        }

    def detect_all_bays(
        self,
        frame: np.ndarray,
        slots: list[dict[str, Any]],
        image_shape: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Run occupancy detection on all bays and return results."""
        results: list[dict[str, Any]] = []
        for slot in slots:
            result = self.detect_bay_occupancy(frame, slot, image_shape)
            results.append(result)
        return results


# Global singleton
_global_bg_detector: ReferenceFrameDetector | None = None


def get_bg_detector() -> ReferenceFrameDetector:
    """Get or create the global ReferenceFrameDetector singleton."""
    global _global_bg_detector
    if _global_bg_detector is None:
        _global_bg_detector = ReferenceFrameDetector()
    return _global_bg_detector
