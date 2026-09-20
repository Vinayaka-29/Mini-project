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
from app.vision.tracker import CentroidTracker
from app.vision.bg_subtractor import get_bg_detector
from app.vision.camera_sources import WebcamSource, PhoneCameraSource, RTSPCameraSource
from app.parking.state_manager import ParkingStateManager
from app.parking.allocation_engine import AllocationEngine
from app.vision.detector import VehicleDetector, get_detector

logger = logging.getLogger(__name__)


class ParkingService:
    def __init__(self) -> None:
        self.state_manager = ParkingStateManager()
        self.detector = get_detector()
        self.bg_detector = get_bg_detector()
        self.tracker = CentroidTracker(max_missing=5, max_distance=80.0)
        self.allocation_engine = AllocationEngine()
        self.camera_active: bool = False
        self.camera_cap: cv2.VideoCapture | None = None
        self.last_frame_annotated_b64: str | None = None
        self.camera_fps: float = 0.0
        self.camera_source_name: str = "WEBCAM"
        # Track consecutive-frame detections for flicker suppression (YOLO mode)
        self._frame_detection_counts: dict[str, int] = {}
        self._stable_frames_required: int = 2

    def initialize_demo_state(self) -> None:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "parking_layout.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            self.state_manager.load_layout(config)
            logger.info("Initialized parking state with %d slots from config.", len(self.state_manager.slots))
        else:
            logger.warning("Config path not found: %s", config_path)

    # ------------------------------------------------------------------
    # Image upload processing
    # ------------------------------------------------------------------

    def process_image_upload(self, file_bytes: bytes) -> dict[str, Any]:
        """Process an uploaded image file and update slot occupancy.

        Uses background subtraction (primary mode) or YOLO (if mode is 'yolo').
        """
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_np = np.array(image)
        frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        image_shape = (h, w)

        start_time = time.time()

        if self.detector.detection_mode == "background_subtraction":
            results = self._process_bg_subtraction(frame_bgr, image_shape)
        else:
            results = self._process_yolo(frame_bgr, image_shape)

        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        # Annotate frame with slot overlays
        annotated_bgr = self.detector.annotate_frame(
            frame_bgr, results.get("detections_obj", []),
            slots=self.state_manager.get_slots(), image_shape=image_shape,
        )
        annotated_b64 = self.detector.encode_base64_jpeg(annotated_bgr)
        self.last_frame_annotated_b64 = annotated_b64

        return {
            "overview": self.state_manager.get_overview(),
            "slots": self.state_manager.get_slots(),
            "detections": results.get("detections", []),
            "total_detected_vehicles": results.get("total_vehicles", 0),
            "inference_time_ms": inference_time_ms,
            "annotated_image": annotated_b64,
            "detection_mode": self.detector.detection_mode,
            "timestamp": time.time(),
        }

    def _process_bg_subtraction(self, frame: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
        """Apply per-bay background subtraction and update state."""
        slots_snapshot = self.state_manager.get_slots()
        bay_results = self.bg_detector.detect_all_bays(frame, slots_snapshot, image_shape)

        occupied_count = 0
        bay_details = []
        for res in bay_results:
            new_status = res["status"]
            conf = res["confidence"]
            self.state_manager.update_slot_status(res["slot_id"], new_status, conf)
            if new_status == "OCCUPIED":
                occupied_count += 1
            bay_details.append({
                "slot_id": res["slot_id"],
                "status": new_status,
                "confidence": conf,
                "mean_diff": res.get("mean_diff", 0.0),
                "edge_ratio": res.get("edge_ratio", 0.0),
            })

        return {
            "detections": bay_details,
            "detections_obj": [],
            "total_vehicles": occupied_count,
        }

    def _process_yolo(self, frame: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
        """YOLO detection path with centroid tracker for flicker suppression."""
        raw_detections = self.detector._detect_yolo(frame, conf_threshold=0.25)

        # Run centroid tracker
        det_dicts = [d.to_dict() for d in raw_detections]
        tracked = self.tracker.update(det_dicts)

        # Update consecutive frame counters (flicker suppression: require ≥2 frames)
        current_ids = {d["tracking_id"] for d in tracked if d.get("tracking_id")}
        new_counts: dict[str, int] = {}
        for det in tracked:
            tid = det.get("tracking_id")
            if tid is not None:
                new_counts[str(tid)] = self._frame_detection_counts.get(str(tid), 0) + 1
        self._frame_detection_counts = new_counts

        stable_dets = [
            d for d in raw_detections
            if self._frame_detection_counts.get(str(d.tracking_id), 0) >= self._stable_frames_required
        ] if raw_detections and raw_detections[0].tracking_id is not None else raw_detections

        self.state_manager.process_vision_detections(stable_dets, image_shape)

        return {
            "detections": [d.to_dict() for d in stable_dets],
            "detections_obj": stable_dets,
            "total_vehicles": len(stable_dets),
        }

    # ------------------------------------------------------------------
    # Live camera stream (MJPEG)
    # ------------------------------------------------------------------

    def generate_camera_stream(self) -> Generator[bytes, None, None]:
        """Yield MJPEG frames for real-time live camera feed."""
        source = WebcamSource(index=0, camera_id="CAM_01")
        use_webcam = source.capture.isOpened()
        if not use_webcam:
            logger.warning("Webcam unavailable — streaming simulated frames.")
            source.release()

        sim_frame_idx = 0
        while True:
            start_t = time.time()

            if use_webcam and self.camera_active:
                vf = source.read()
                frame = vf.frame if vf else None
                if frame is None:
                    frame = self._generate_simulated_frame(sim_frame_idx)
                    sim_frame_idx += 1
            else:
                frame = self._generate_simulated_frame(sim_frame_idx)
                sim_frame_idx += 1

            h, w = frame.shape[:2]

            if self.camera_active:
                if self.detector.detection_mode == "background_subtraction":
                    self._process_bg_subtraction(frame, (h, w))
                else:
                    self._process_yolo(frame, (h, w))

            annotated = self.detector.annotate_frame(
                frame, [],
                slots=self.state_manager.get_slots(),
                image_shape=(h, w),
            )

            elapsed = time.time() - start_t
            if elapsed > 0:
                self.camera_fps = round(1.0 / elapsed, 1)

            cv2.putText(
                annotated,
                f"FPS: {self.camera_fps} | {self.detector.detection_mode.upper()} | {'LIVE' if self.camera_active else 'STANDBY'}",
                (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )

            success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.04)  # ~25 FPS target

    def _generate_simulated_frame(self, frame_idx: int) -> np.ndarray:
        """Create a synthetic parking scene for when no camera is available."""
        w, h = 640, 360
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (26, 32, 44)

        cv2.rectangle(img, (50, 40), (590, 280), (35, 45, 60), -1)
        cv2.line(img, (60, 160), (580, 160), (100, 110, 130), 2, cv2.LINE_AA)

        for x in range(100, 560, 115):
            cv2.line(img, (x, 50), (x, 140), (70, 80, 95), 1)
            cv2.line(img, (x, 175), (x, 265), (70, 80, 95), 1)

        scan_x = int((frame_idx * 4) % w)
        cv2.line(img, (scan_x, 40), (scan_x, 280), (0, 255, 157), 1)

        for slot in self.state_manager.get_slots():
            if slot.get("status") == "OCCUPIED":
                raw_poly = slot.get("polygon", [])
                if raw_poly and len(raw_poly) >= 4:
                    from app.vision.occupancy import scale_polygon
                    poly = scale_polygon(raw_poly, (h, w))
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    cv2.rectangle(img, (cx - 30, cy - 18), (cx + 30, cy + 18), (180, 50, 60), -1)
                    cv2.rectangle(img, (cx - 18, cy - 12), (cx + 18, cy + 12), (220, 220, 240), -1)

        return img

    # ------------------------------------------------------------------
    # Reference frame management (BG subtraction setup)
    # ------------------------------------------------------------------

    def set_reference_frame(self, file_bytes: bytes) -> dict[str, Any]:
        """Upload and set the empty-lot reference frame for background subtraction."""
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            img_np = np.array(image)
            frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            self.bg_detector.set_reference(frame_bgr)
            return {
                "status": "success",
                "message": "Reference frame set successfully. Background subtraction is now calibrated.",
                "resolution": f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}",
                "has_reference": True,
            }
        except Exception as exc:
            logger.error("Failed to set reference frame: %s", exc, exc_info=True)
            return {"status": "error", "message": str(exc), "has_reference": False}

    # ------------------------------------------------------------------
    # Convenience delegation methods
    # ------------------------------------------------------------------

    def toggle_camera(self, active: bool | None = None) -> dict[str, Any]:
        if active is not None:
            self.camera_active = active
        else:
            self.camera_active = not self.camera_active
        return {
            "camera_active": self.camera_active,
            "status": "RUNNING" if self.camera_active else "STOPPED",
            "source": self.camera_source_name,
            "detection_mode": self.detector.detection_mode,
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
        """Allocate nearest available slot via AllocationEngine, then reserve it."""
        available = [s for s in self.state_manager.get_slots() if s["status"] == "AVAILABLE"]
        if not available:
            raise ValueError("Parking Full! No available slots currently.")
        best = self.allocation_engine.select_best_slot(available, vehicle_type)
        return self.state_manager.allocate_slot(vehicle_id, vehicle_type, preferred_slot_id=best["slot_id"])

    def get_system_status(self) -> dict[str, Any]:
        overview = self.state_manager.get_overview()
        return {
            "status": "healthy",
            "camera_active": self.camera_active,
            "camera_fps": self.camera_fps,
            "detection_mode": self.detector.detection_mode,
            "bg_subtraction_calibrated": self.bg_detector.has_reference,
            "model": "Background Subtraction (primary) / YOLOv8 (optional)",
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
