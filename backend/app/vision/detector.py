from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Vehicle classes in COCO dataset: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Detection mode: "background_subtraction" (default, reliable for fixed cam)
# or "yolo" (requires domain-appropriate model)
DETECTION_MODE = "background_subtraction"


@dataclass
class VehicleDetection:
    tracking_id: int | None = None
    class_name: str = "car"
    confidence: float = 0.0
    bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # [x1, y1, x2, y2]
    camera_id: str = "CAM_01"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracking_id": self.tracking_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(coord, 1) for coord in self.bbox],
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
        }


class VehicleDetector:
    """Vehicle detector supporting background subtraction (primary) and YOLOv8 (optional).

    Background subtraction is the default and recommended mode for fixed-camera
    miniature parking lots.  YOLO is kept as an optional path and requires a
    domain-appropriate fine-tuned model to be useful on cardboard/toy lots.
    (Thread-safe Singleton)
    """

    _instance: "VehicleDetector | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, model_name: str = "yolov8n.pt", camera_id: str = "CAM_01") -> "VehicleDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "yolov8n.pt", camera_id: str = "CAM_01") -> None:
        if getattr(self, "_initialized", False):
            return
        self.camera_id = camera_id
        self.model_name = model_name
        self.model = None
        self.detection_mode = DETECTION_MODE
        # Lazy-load YOLO only if explicitly requested
        if self.detection_mode == "yolo":
            self._load_model()
        self._initialized = True

    def set_detection_mode(self, mode: str) -> None:
        """Switch detection mode: 'background_subtraction' or 'yolo'."""
        if mode not in ("background_subtraction", "yolo"):
            raise ValueError(f"Unknown detection mode: {mode!r}. Use 'background_subtraction' or 'yolo'.")
        self.detection_mode = mode
        if mode == "yolo" and self.model is None:
            self._load_model()
        logger.info("Detection mode switched to: %s", mode)

    def _load_model(self) -> None:
        """Explicitly locate and load YOLOv8 weights with critical error reporting."""
        try:
            from ultralytics import YOLO

            model_target = self.model_name
            candidate_paths = [
                Path.cwd() / self.model_name,
                Path(__file__).resolve().parents[2] / self.model_name,
                Path(__file__).resolve().parents[3] / self.model_name,
            ]
            for path in candidate_paths:
                if path.exists():
                    model_target = str(path)
                    logger.info("Found local YOLO weights file at: %s", model_target)
                    break

            self.model = YOLO(model_target)
            logger.info("Successfully loaded YOLO model: %s", model_target)
        except Exception as exc:
            logger.critical("CRITICAL: Failed to load YOLO model (%s): %s", self.model_name, exc, exc_info=True)
            self.model = None

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> list[VehicleDetection]:
        """Detect vehicles. Routes to background subtraction or YOLO based on detection_mode.

        NOTE: Background subtraction mode returns an empty list here — occupancy is computed
        per-bay directly via ReferenceFrameDetector.detect_bay_occupancy() in parking_service.
        This method is the YOLO path kept for compatibility.
        """
        if self.detection_mode == "background_subtraction":
            # BG subtraction works at the bay level, not as a detection list.
            # Return empty — occupancy is determined in parking_service/state_manager.
            return []

        return self._detect_yolo(frame, conf_threshold)

    def _detect_yolo(self, frame: np.ndarray, conf_threshold: float = 0.25) -> list[VehicleDetection]:
        """Run YOLOv8 inference and filter for vehicle classes only."""
        if frame is None or frame.size == 0:
            return []

        # Robust image preprocessing: ensure 3-channel BGR format
        if len(frame.shape) == 2:
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            processed_frame = frame

        if self.model is None:
            self._load_model()
            if self.model is None:
                logger.error("YOLO model unavailable for detection.")
                return []

        detections: list[VehicleDetection] = []

        try:
            results = self.model.predict(
                source=processed_frame,
                conf=conf_threshold,
                imgsz=640,
                verbose=False,
            )
            if results and len(results) > 0:
                first_result = results[0]
                boxes = first_result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                        if cls_id in VEHICLE_CLASS_IDS:
                            conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                            class_name = VEHICLE_CLASS_IDS.get(cls_id, "car")
                            detections.append(
                                VehicleDetection(
                                    class_name=class_name,
                                    confidence=conf,
                                    bbox=xyxy,
                                    camera_id=self.camera_id,
                                )
                            )
        except Exception as exc:
            logger.error("YOLO detection execution error: %s", exc, exc_info=True)

        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[VehicleDetection],
        slots: list[dict[str, Any]] | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Draw bounding boxes (YOLO mode) or slot overlays (BG subtraction mode)."""
        from app.vision.occupancy import scale_polygon

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw slot polygon overlays with status colours
        if slots:
            for slot in slots:
                raw_poly = slot.get("polygon", [])
                if not raw_poly or len(raw_poly) < 3:
                    continue
                poly = scale_polygon(raw_poly, image_shape or (h, w))
                pts = np.array([[int(p[0]), int(p[1])] for p in poly], dtype=np.int32)
                status = slot.get("status", "AVAILABLE")
                color = (0, 255, 157) if status == "AVAILABLE" else (255, 0, 85) if status == "OCCUPIED" else (255, 183, 0)
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
                # Semi-transparent fill
                overlay = annotated.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
                # Bay ID label
                cx = int(np.mean([p[0] for p in poly]))
                cy = int(np.mean([p[1] for p in poly]))
                cv2.putText(annotated, slot.get("slot_id", "?"), (cx - 12, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw YOLO bounding boxes if in YOLO mode
        for idx, det in enumerate(detections):
            bbox = det.bbox
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = (255, 0, 85) if idx % 2 == 0 else (0, 240, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            tag = f"AI {det.class_name.upper()} {int(det.confidence * 100)}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 18)), (x1 + tw + 6, max(18, y1)), color, -1)
            cv2.putText(annotated, tag, (x1 + 3, max(13, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        # HUD banner
        mode_label = "BG-SUB" if self.detection_mode == "background_subtraction" else "YOLOv8"
        cv2.rectangle(annotated, (10, 10), (min(w - 10, 460), 44), (7, 10, 18), -1)
        cv2.rectangle(annotated, (10, 10), (min(w - 10, 460), 44), (0, 240, 255), 1)
        cv2.putText(
            annotated,
            f"AI-PARK {mode_label} | SLOTS: {len(slots) if slots else 0}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 157),
            1,
            cv2.LINE_AA,
        )

        return annotated

    def encode_base64_jpeg(self, image: np.ndarray) -> str:
        """Encode OpenCV image to base64 JPEG data URL string."""
        success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            return ""
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


# Global persistent singleton instance for reuse across endpoints
global_detector = VehicleDetector()


def get_detector() -> VehicleDetector:
    """Helper function to get the global singleton VehicleDetector instance."""
    return global_detector
