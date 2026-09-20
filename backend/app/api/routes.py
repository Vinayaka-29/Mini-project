from __future__ import annotations

import logging
from typing import Any, Optional, Union
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.services.parking_service import parking_service
from app.vision.detector import get_detector
from app.vision.occupancy import global_occupancy_engine, calculate_box_iou
from app.vision.gemini_calibrator import calibrate_image

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Request models ─────────────────────────────────────────────────────────────

class AllocateRequest(BaseModel):
    vehicle_id: str
    vehicle_type: str = "car"


class CameraToggleRequest(BaseModel):
    active: Optional[bool] = None


class DetectionModeRequest(BaseModel):
    mode: str  # "background_subtraction" or "yolo"


class SlotLayoutItem(BaseModel):
    slot_id: str
    polygon: list[list[float]]
    section_id: Optional[str] = "A"
    camera_id: Optional[str] = "CAM_01"
    status: Optional[str] = "AVAILABLE"
    type: Optional[str] = "STANDARD"
    priority: Optional[int] = 1
    distance_from_entries: Optional[float] = 0.0


class LayoutUpdateRequest(BaseModel):
    camera_id: Optional[str] = "CAM_01"
    section_id: Optional[str] = "A"
    slots: list[dict[str, Any]]


# ── Core endpoints ─────────────────────────────────────────────────────────────

@router.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "ai-park",
        "detection_mode": get_detector().detection_mode,
        "bg_calibrated": parking_service.bg_detector.has_reference,
    }


@router.get("/overview")
async def overview() -> dict[str, Any]:
    return parking_service.get_overview()


@router.get("/slots")
async def slots() -> dict[str, Any]:
    return {"slots": parking_service.get_slots()}


@router.get("/sections")
async def sections() -> dict[str, Any]:
    return {"sections": parking_service.get_sections()}


@router.get("/allocations")
async def allocations() -> dict[str, Any]:
    return {"allocations": parking_service.state_manager.get_allocations()}


@router.get("/vehicles")
async def vehicles() -> dict[str, Any]:
    return {"vehicles": parking_service.state_manager.get_vehicles()}


@router.get("/cameras")
async def cameras() -> dict[str, Any]:
    camera_ids = sorted({slot["camera_id"] for slot in parking_service.state_manager.slots.values()})
    return {"cameras": [{"camera_id": camera_id, "status": "ONLINE"} for camera_id in camera_ids]}


@router.get("/events")
async def events() -> dict[str, Any]:
    return {"events": parking_service.get_recent_events()}


@router.post("/allocate")
async def allocate(payload: AllocateRequest) -> dict[str, Any]:
    try:
        result = parking_service.allocate_slot(payload.vehicle_id, payload.vehicle_type)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/system")
async def system() -> dict[str, Any]:
    return parking_service.get_system_status()


# ── Layout configuration ───────────────────────────────────────────────────────

@router.post("/config/layout")
async def update_layout(
    payload: Union[LayoutUpdateRequest, dict[str, Any], list[dict[str, Any]]]
) -> dict[str, Any]:
    """Dynamically update parking spot polygons in memory and disk."""
    try:
        raw_layout: dict[str, Any] | list[dict[str, Any]]
        if isinstance(payload, LayoutUpdateRequest):
            raw_layout = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        elif isinstance(payload, (dict, list)):
            raw_layout = payload
        else:
            raw_layout = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)

        result = parking_service.state_manager.update_layout(raw_layout)
        global_occupancy_engine.set_layout(raw_layout)

        logger.info("Dynamic layout updated: %d slots loaded", result.get("total_slots", 0))
        return result
    except Exception as exc:
        logger.error("Failed to update layout: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to update layout: {str(exc)}") from exc


# ── Reference frame (BG subtraction calibration) ──────────────────────────────

@router.post("/reference-frame")
async def set_reference_frame(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload an empty-lot photo to calibrate background subtraction.

    This must be called once before background-subtraction detection works.
    Use a photo taken from the fixed camera with NO vehicles present.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image file received")

    def _set(img_bytes: bytes) -> dict[str, Any]:
        return parking_service.set_reference_frame(img_bytes)

    result = await run_in_threadpool(_set, content)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Failed to set reference frame"))
    return result


@router.get("/reference-frame/status")
async def reference_frame_status() -> dict[str, Any]:
    return {
        "has_reference": parking_service.bg_detector.has_reference,
        "detection_mode": get_detector().detection_mode,
    }


# ── Detection mode switching ───────────────────────────────────────────────────

@router.post("/config/detection-mode")
async def set_detection_mode(payload: DetectionModeRequest) -> dict[str, Any]:
    """Switch between 'background_subtraction' (default) and 'yolo' detection."""
    try:
        get_detector().set_detection_mode(payload.mode)
        return {"status": "success", "detection_mode": payload.mode}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Image detection ────────────────────────────────────────────────────────────

@router.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0, description="YOLO confidence threshold (yolo mode only)"),
) -> dict[str, Any]:
    """Process an uploaded image. Routes to background subtraction or YOLO depending on mode.

    Returns updated slot states, overview, and annotated image (base64).
    All status values come from real detection — no mocked data.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image file received")

    def pipeline(img_bytes: bytes) -> dict[str, Any]:
        import numpy as np
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image (invalid or corrupted file)")
        return parking_service.process_image_upload(img_bytes)

    try:
        return await run_in_threadpool(pipeline, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Inference failed",
                "detail": str(exc),
                "slots": parking_service.state_manager.get_slots(),
                "overview": parking_service.state_manager.get_overview(),
            },
        )


# ── Camera streaming ───────────────────────────────────────────────────────────

@router.post("/camera/toggle")
async def toggle_camera(payload: CameraToggleRequest = CameraToggleRequest()) -> dict[str, Any]:
    """Start or stop the camera video feed & continuous AI scanning."""
    return parking_service.toggle_camera(active=payload.active)


@router.get("/camera/feed")
async def camera_feed():
    """Live MJPEG video stream with real-time detections and slot overlays."""
    return StreamingResponse(
        parking_service.generate_camera_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Simulation / testing ───────────────────────────────────────────────────────

@router.post("/simulate/event")
async def simulate_event() -> dict[str, Any]:
    """Trigger a simulated vehicle arrival or departure (testing only)."""
    return parking_service.simulate_random_event()


@router.post("/simulate/reset")
async def simulate_reset() -> dict[str, Any]:
    """Reset all parking bays to available."""
    return parking_service.reset_slots()


@router.post("/detect/auto")
async def detect_auto(file: UploadFile = File(...)) -> dict[str, Any]:
    """Auto-calibrate parking bays using Gemini and detect vehicle occupancy with YOLO."""
    image_bytes = await file.read()

    try:
        bays = await run_in_threadpool(calibrate_image, image_bytes, file.content_type or "image/jpeg")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Calibration failed: {e}")
    except Exception as e:
        logger.error("Unexpected calibration error: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Calibration failed: {type(e).__name__}: {e}")

    if not bays:
        return {"bays": [], "vehicles": [], "message": "No parking bays detected in this image."}

    def process_detection():
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image (invalid or corrupted file)")

        h, w = frame.shape[:2]

        detector = get_detector()
        detector.set_detection_mode("yolo")

        try:
            vehicle_detections = detector._detect_yolo(frame, conf_threshold=0.25)
        except TypeError:
            # Signature might not accept conf_threshold as a kwarg — retry positionally,
            # then with no confidence arg at all.
            try:
                vehicle_detections = detector._detect_yolo(frame, 0.25)
            except TypeError:
                vehicle_detections = detector._detect_yolo(frame)

        vehicles = []
        for det in vehicle_detections:
            # Handle both object-style (det.bbox) and dict-style (det["bbox"]) detections
            if isinstance(det, dict):
                bbox = det.get("bbox") or det.get("box")
                class_name = det.get("class_name") or det.get("class") or det.get("label", "vehicle")
                confidence = det.get("confidence", 0.0)
            else:
                bbox = getattr(det, "bbox", None) or getattr(det, "box", None)
                class_name = getattr(det, "class_name", None) or getattr(det, "label", "vehicle")
                confidence = getattr(det, "confidence", 0.0)

            if bbox is None:
                raise ValueError(f"Could not extract bounding box from detection object: {det!r}")

            # Normalize to 0-1. If values already look normalized (all <= 1.0), leave as-is.
            if all(0.0 <= v <= 1.0 for v in bbox):
                norm_box = list(bbox)
            else:
                norm_box = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

            vehicles.append({
                "class": class_name,
                "confidence": confidence,
                "box": norm_box
            })

        results = []
        for bay in bays:
            bx = bay["box"]
            best_overlap = 0.0
            for v in vehicles:
                try:
                    overlap = calculate_box_iou(bx, v["box"])
                except Exception as e:
                    raise ValueError(f"calculate_box_iou failed on bay={bx} vehicle={v['box']}: {type(e).__name__}: {e}")
                best_overlap = max(best_overlap, overlap)
            status = "occupied" if best_overlap >= 0.4 else "free"
            results.append({
                "id": bay["id"],
                "box": bx,
                "status": status,
                "overlap": round(best_overlap, 3)
            })

        return {"bays": results, "vehicles": vehicles}

    try:
        return await run_in_threadpool(process_detection)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Detection failed: {e}")
    except Exception as e:
        logger.error("detect_auto pipeline failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {type(e).__name__}: {e}")