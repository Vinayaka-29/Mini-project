from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Vehicle classes in COCO: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


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
    """YOLOv8-powered real-time vehicle detector (Thread-safe Singleton)."""

    _instance: VehicleDetector | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, model_name: str = "yolov8n.pt", camera_id: str = "CAM_01") -> VehicleDetector:
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
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            logger.info("Successfully loaded YOLO model: %s", self.model_name)
        except Exception as exc:
            logger.warning("Could not load YOLO model (%s). Will use fallback/simulated mode: %s", self.model_name, exc)
            self.model = None

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.15) -> list[VehicleDetection]:
        """Detect vehicles in an image frame (BGR format) using YOLOv8."""
        if frame is None or frame.size == 0:
            return []

        if self.model is None:
            self._load_model()

        detections: list[VehicleDetection] = []

        if self.model is not None:
            try:
                results = self.model.predict(source=frame, conf=conf_threshold, imgsz=640, verbose=False)
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
                logger.error("YOLO detection error: %s", exc)

        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[VehicleDetection],
        slots: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Draw accurate bounding boxes around detected vehicles with HUD telemetry."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw detected vehicle bounding boxes directly where cars are located
        for idx, det in enumerate(detections):
            bbox = det.bbox
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # High-tech neon bounding box
            color = (255, 0, 85) if idx % 2 == 0 else (0, 240, 255)  # Crimson / Cyan
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw transparent box highlight
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

            # Tag label background
            tag = f"AI {det.class_name.upper()} {int(det.confidence * 100)}%"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 18)), (x1 + tw + 6, max(18, y1)), color, -1)
            cv2.putText(
                annotated,
                tag,
                (x1 + 3, max(13, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Futuristic HUD Header banner
        cv2.rectangle(annotated, (10, 10), (min(w - 10, 420), 44), (7, 10, 18), -1)
        cv2.rectangle(annotated, (10, 10), (min(w - 10, 420), 44), (0, 240, 255), 1)
        cv2.putText(
            annotated,
            f"AI-PARK YOLOv8 | VEHICLES DETECTED: {len(detections)}",
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
