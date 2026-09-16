import io
import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.vision.occupancy import (
    SlotOccupancyEngine,
    load_parking_layout,
    is_point_in_polygon,
    calculate_box_iou,
    calculate_polygon_bbox_overlap,
)
from app.vision.detector import VehicleDetection, VehicleDetector, get_detector


def test_vehicle_detector_singleton():
    """Verify VehicleDetector adheres to global singleton pattern."""
    detector1 = VehicleDetector()
    detector2 = VehicleDetector()
    detector3 = get_detector()
    assert detector1 is detector2
    assert detector2 is detector3


def test_parking_layout_loader():
    """Verify loading parking spots from configs/parking_layout.json."""
    slots = load_parking_layout()
    assert len(slots) > 0
    assert "slot_id" in slots[0]
    assert "polygon" in slots[0]


def test_geometric_overlap_and_point_in_polygon():
    """Test point-in-polygon and polygon-bbox overlap calculations."""
    poly = [[100.0, 50.0], [200.0, 50.0], [200.0, 150.0], [100.0, 150.0]]
    assert is_point_in_polygon((150, 100), poly) is True
    assert is_point_in_polygon((50, 50), poly) is False

    # Calculate bbox overlap (>50%)
    bbox_overlap = [110.0, 60.0, 190.0, 140.0]
    overlap_ratio, iou, center_inside = calculate_polygon_bbox_overlap(poly, bbox_overlap)
    assert overlap_ratio > 0.50
    assert center_inside is True

    # Box IoU
    iou_val = calculate_box_iou([0, 0, 10, 10], [0, 0, 10, 10])
    assert iou_val == 1.0


def test_occupancy_engine_detection_overlap():
    engine = SlotOccupancyEngine(iou_threshold=0.20)
    slot = {
        "slot_id": "A01",
        "polygon": [[100, 50], [200, 50], [200, 150], [100, 150]],
        "status": "AVAILABLE",
    }

    # Overlapping vehicle detection bbox (> 50% overlap)
    det = VehicleDetection(
        class_name="car",
        confidence=0.95,
        bbox=[110, 60, 190, 140],
    )

    res = engine.compute_slot_status(slot, [det])
    assert res["status"] == "OCCUPIED"
    assert res["confidence"] >= 0.80

    # Non-overlapping vehicle
    det_outside = VehicleDetection(
        class_name="car",
        confidence=0.95,
        bbox=[400, 400, 500, 500],
    )
    res_empty = engine.compute_slot_status(slot, [det_outside])
    assert res_empty["status"] == "AVAILABLE"


def test_detector_annotation_runs_smoothly():
    detector = get_detector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    slots = [
        {"slot_id": "A01", "polygon": [[50, 50], [150, 50], [150, 150], [50, 150]], "status": "AVAILABLE"}
    ]
    annotated = detector.annotate_frame(dummy_frame, [], slots)
    assert annotated.shape == (480, 640, 3)
    b64 = detector.encode_base64_jpeg(annotated)
    assert b64.startswith("data:image/jpeg;base64,")


def test_image_upload_endpoint():
    """Verify non-blocking image upload endpoint processes image and returns updated parking lot state."""
    client = TestClient(app)
    
    # Create test dummy image in memory
    img = Image.new("RGB", (640, 480), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post("/api/detect/image", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "overview" in data
    assert "slots" in data
    assert "detections" in data
    assert "annotated_image" in data
    assert "total_detected_vehicles" in data
    assert "inference_time_ms" in data
