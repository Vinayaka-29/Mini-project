from __future__ import annotations

import logging
from typing import Any, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.parking_service import parking_service

logger = logging.getLogger(__name__)

router = APIRouter()


class AllocateRequest(BaseModel):
    vehicle_id: str
    vehicle_type: str = "car"


class CameraToggleRequest(BaseModel):
    active: Optional[bool] = None


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


@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Non-blocking endpoint for uploaded image processing:
    1. Receives uploaded image asynchronously.
    2. Runs YOLOv8 vehicle detection on worker threadpool via global VehicleDetector.
    3. Calculates spot availability using IoU and Point-in-Polygon occupancy engine.
    4. Updates parking state in state_manager.
    5. Returns updated parking lot JSON state to the frontend.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty image file received")

        # Execute the CPU/GPU-bound computer vision pipeline on non-blocking threadpool
        result = await run_in_threadpool(parking_service.process_image_upload, content)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Image detection failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image detection failed: {str(exc)}") from exc


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
