from __future__ import annotations

import logging
from typing import Any, Optional, Union
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.services.parking_service import parking_service
from app.vision.detector import get_detector
from app.vision.occupancy import global_occupancy_engine

logger = logging.getLogger(__name__)

router = APIRouter()


class AllocateRequest(BaseModel):
    vehicle_id: str
    vehicle_type: str = "car"


class CameraToggleRequest(BaseModel):
    active: Optional[bool] = None


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


@router.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "ai-park", "vision_model": "YOLOv8-Nano"}


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


@router.post("/config/layout")
async def update_layout(payload: Union[LayoutUpdateRequest, dict[str, Any], list[dict[str, Any]]]) -> dict[str, Any]:
    """
    Dynamically update parking spot polygons in memory.
    Accepts new polygon coordinates from frontend or custom camera calibrations.
    """
    try:
        raw_layout: dict[str, Any] | list[dict[str, Any]]
        if isinstance(payload, LayoutUpdateRequest):
            raw_layout = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        elif isinstance(payload, (dict, list)):
            raw_layout = payload
        else:
            raw_layout = payload.model_dump() if hasattr(payload, "model_dump") else (payload.dict() if hasattr(payload, "dict") else dict(payload))


        # 1. Update state manager slots in memory
        result = parking_service.state_manager.update_layout(raw_layout)

        # 2. Update occupancy engine default slots
        global_occupancy_engine.set_layout(raw_layout)

        logger.info("Dynamic layout successfully updated: %d slots loaded", result.get("total_slots", 0))
        return result
    except Exception as exc:
        logger.error("Failed to update layout dynamically: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to update layout: {str(exc)}") from exc


@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Non-blocking endpoint for uploaded image processing with graceful inference fallback:
    1. Receives uploaded image.
    2. Runs YOLOv8 vehicle detection in worker threadpool.
    3. Calculates spot availability using IoU and Point-in-Polygon occupancy engine.
    4. Saves state and returns updated parking lot state.
    5. Gracefully catches inference / OOM errors and returns 500 JSON without crashing.
    """
    try:
        content = await file.read()
        if not content:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty image file received", "slots": []},
            )

        def pipeline(img_bytes: bytes) -> dict[str, Any]:
            # 1. Convert bytes to OpenCV image
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode image from uploaded bytes")

            # 2. Get global detector & find cars
            detector = get_detector()
            detections = detector.detect(frame)

            # 3. Calculate intersections
            current_slots = list(parking_service.state_manager.slots.values())
            h, w = frame.shape[:2]
            updated = global_occupancy_engine.map_detections_to_spots(
                slots=current_slots, detections=detections, image_shape=(h, w)
            )

            # 4. Save state
            for spot in updated:
                parking_service.state_manager.update_slot_status(spot["slot_id"], spot["status"])

            # 5. Return updated lot
            return {
                "slots": parking_service.state_manager.get_slots(),
                "overview": parking_service.state_manager.get_overview(),
                "detections": [d.to_dict() if hasattr(d, "to_dict") else d for d in detections],
                "total_detected_vehicles": len(detections),
            }

        return await run_in_threadpool(pipeline, content)

    except Exception as exc:
        logger.error("Inference execution failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Inference failed", "detail": str(exc), "slots": []},
        )


@router.post("/camera/toggle")
async def toggle_camera(payload: CameraToggleRequest = CameraToggleRequest()) -> dict[str, Any]:
    """Start or stop the camera video feed & continuous AI scanning."""
    return parking_service.toggle_camera(active=payload.active)


@router.get("/camera/feed")
async def camera_feed():
    """Live MJPEG video stream with real-time YOLO detections and slot overlays."""
    return StreamingResponse(
        parking_service.generate_camera_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/simulate/event")
async def simulate_event() -> dict[str, Any]:
    """Trigger a simulated vehicle arrival or departure."""
    return parking_service.simulate_random_event()


@router.post("/simulate/reset")
async def simulate_reset() -> dict[str, Any]:
    """Reset all parking bays to available."""
    return parking_service.reset_slots()
