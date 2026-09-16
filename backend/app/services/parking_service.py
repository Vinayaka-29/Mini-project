from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Generator
import cv2
import numpy as np
from PIL import Image

from app.vision.occupancy import SlotOccupancyEngine

from app.parking.state_manager import ParkingStateManager
from app.vision.detector import VehicleDetector, get_detector

logger = logging.getLogger(__name__)


class ParkingService:
    def __init__(self) -> None:
        self.state_manager = ParkingStateManager()
        self.detector = get_detector()
        self.camera_active: bool = False
        self.camera_cap: cv2.VideoCapture | None = None
        self.last_frame_annotated_b64: str | None = None
        self.camera_fps: float = 0.0
        self.camera_source_name: str = "WEBCAM / SIMULATED"

    def initialize_demo_state(self) -> None:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "parking_layout.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            self.state_manager.load_layout(config)
        else:
            logger.warning("Config path not found: %s", config_path)

    def process_image_upload(self, file_bytes: bytes) -> dict[str, Any]:
        """Process an uploaded image file with YOLOv8 and update slot occupancy."""
        # Convert bytes to OpenCV BGR image
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_np = np.array(image)
        frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        h, w = frame_bgr.shape[:2]

        # Run YOLO detection
        start_time = time.time()
        detections = self.detector.detect(frame_bgr, conf_threshold=0.25)
        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        # Update slot occupancy states
        self.state_manager.process_vision_detections(detections, image_shape=(h, w))

        # Annotate image with real YOLO detected vehicle bounding boxes ONLY
        annotated_bgr = self.detector.annotate_frame(frame_bgr, detections, slots=[])
        annotated_b64 = self.detector.encode_base64_jpeg(annotated_bgr)
        self.last_frame_annotated_b64 = annotated_b64

        return {
            "overview": self.state_manager.get_overview(),
            "slots": self.state_manager.get_slots(),
            "detections": [d.to_dict() for d in detections],
            "total_detected_vehicles": len(detections),
            "inference_time_ms": inference_time_ms,
            "annotated_image": annotated_b64,
            "timestamp": time.time(),
        }

    def generate_camera_stream(self) -> Generator[bytes, None, None]:
        """Yield MJPEG frames for real-time live camera feed."""
        # Try local webcam first; if unavailable, generate high-tech simulated CCTV feed
        cap = cv2.VideoCapture(0)
        use_webcam = cap.isOpened()

        sim_frame_idx = 0
        while True:
            start_t = time.time()
            if use_webcam and self.camera_active:
                ret, frame = cap.read()
                if not ret:
                    frame = self._generate_simulated_frame(sim_frame_idx)
            else:
                frame = self._generate_simulated_frame(sim_frame_idx)
                sim_frame_idx += 1

            # Run detection periodically or on each frame
            detections = self.detector.detect(frame, conf_threshold=0.30)
            if self.camera_active:
                self.state_manager.process_vision_detections(detections, image_shape=frame.shape[:2])

            annotated = self.detector.annotate_frame(
                frame, detections, self.state_manager.get_slots()
            )

            # Calculate FPS
            elapsed = time.time() - start_t
            if elapsed > 0:
                self.camera_fps = round(1.0 / elapsed, 1)

            cv2.putText(
                annotated,
                f"FPS: {self.camera_fps} | STATUS: {'LIVE SCANNING' if self.camera_active else 'STANDBY'}",
                (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.04)  # ~25 FPS

    def _generate_simulated_frame(self, frame_idx: int) -> np.ndarray:
        """Create a synthetic high-tech miniature parking scene for demonstration."""
        w, h = 640, 360
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Background dark asphalt
        img[:] = (26, 32, 44)

        # Draw parking lot surface and lanes
        cv2.rectangle(img, (50, 40), (590, 280), (35, 45, 60), -1)
        # Lane divider
        cv2.line(img, (60, 150), (580, 150), (100, 110, 130), 2, cv2.LINE_AA)

        # Draw grid lines
        for x in range(100, 550, 90):
            cv2.line(img, (x, 50), (x, 130), (70, 80, 95), 1)
            cv2.line(img, (x, 170), (x, 250), (70, 80, 95), 1)

        # Animated roving scan beam
        scan_x = int((frame_idx * 4) % w)
        cv2.line(img, (scan_x, 40), (scan_x, 280), (0, 255, 157), 1)

        # Draw simulated cars inside occupied slots
        for slot in self.state_manager.get_slots():
            if slot["status"] == "OCCUPIED":
                raw_poly = slot.get("polygon", [])
                if raw_poly and len(raw_poly) >= 4:
                    xs = [p[0] for p in raw_poly]
                    ys = [p[1] for p in raw_poly]
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    # Draw car chassis
                    cv2.rectangle(img, (cx - 30, cy - 18), (cx + 30, cy + 18), (180, 50, 60), -1)
                    cv2.rectangle(img, (cx - 18, cy - 12), (cx + 18, cy + 12), (220, 220, 240), -1)
                    # Headlights
                    cv2.circle(img, (cx + 28, cy - 10), 3, (0, 255, 255), -1)
                    cv2.circle(img, (cx + 28, cy + 10), 3, (0, 255, 255), -1)

        return img

    def toggle_camera(self, active: bool | None = None) -> dict[str, Any]:
        if active is not None:
            self.camera_active = active
        else:
            self.camera_active = not self.camera_active
        return {
            "camera_active": self.camera_active,
            "status": "RUNNING" if self.camera_active else "STOPPED",
            "source": self.camera_source_name,
        }

    def get_overview(self) -> dict[str, Any]:
        return self.state_manager.get_overview()

    def get_slots(self) -> list[dict[str, Any]]:
        return self.state_manager.get_slots()

    def get_recent_events(self) -> list[dict[str, Any]]:
        return self.state_manager.get_recent_events()

    def simulate_random_event(self) -> dict[str, Any]:
        return self.state_manager.simulate_random_change()

    def reset_slots(self) -> dict[str, Any]:
        return self.state_manager.reset_all_slots()

    def allocate_slot(self, vehicle_id: str, vehicle_type: str = "car") -> dict[str, Any]:
        return self.state_manager.allocate_slot(vehicle_id, vehicle_type)

    def get_system_status(self) -> dict[str, Any]:
        overview = self.state_manager.get_overview()
        return {
            "status": "healthy",
            "camera_active": self.camera_active,
            "camera_fps": self.camera_fps,
            "model": "YOLOv8-Nano (COCO Pre-trained)",
            "camera_count": 1,
            "total_slots": overview["total_slots"],
            "occupied_slots": overview["occupied"],
            "available_slots": overview["available"],
            "occupancy_pct": overview["occupancy_pct"],
        }

    def get_sections(self) -> list[dict[str, Any]]:
        sections: dict[str, dict[str, Any]] = {}
        for slot in self.state_manager.get_slots():
            sec = slot.get("section_id", "A")
            if sec not in sections:
                sections[sec] = {"section_id": sec, "total": 0, "available": 0, "occupied": 0}
            sections[sec]["total"] += 1
            if slot.get("status") == "AVAILABLE":
                sections[sec]["available"] += 1
            elif slot.get("status") == "OCCUPIED":
                sections[sec]["occupied"] += 1
        return list(sections.values())



parking_service = ParkingService()

