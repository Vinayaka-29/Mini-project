"""
Vision + occupancy tests updated for:
- Normalized polygon coordinate system (0.0-1.0)
- scale_polygon() and _is_normalized() helpers
- BG subtraction detector path
- A1-B4 bay IDs
- New /api/reference-frame endpoint
"""
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
    scale_polygon,
    _is_normalized,
)
from app.vision.detector import VehicleDetection, VehicleDetector, get_detector
from app.vision.bg_subtractor import ReferenceFrameDetector


# ── Unit tests: geometry ──────────────────────────────────────────────────────

def test_scale_polygon_normalized():
    """Normalized polygon scaled correctly by image dimensions."""
    raw = [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]]
    scaled = scale_polygon(raw, (480, 640))
    assert abs(scaled[0][0] - 64.0) < 0.01   # 0.1 * 640
    assert abs(scaled[0][1] - 96.0) < 0.01   # 0.2 * 480


def test_scale_polygon_absolute():
    """Absolute pixel polygon is returned as-is."""
    raw = [[100.0, 50.0], [200.0, 50.0], [200.0, 150.0], [100.0, 150.0]]
    scaled = scale_polygon(raw, (480, 640))
    assert scaled[0][0] == 100.0


def test_is_normalized_true():
    poly = [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0]]
    assert _is_normalized(poly) is True


def test_is_normalized_false():
    poly = [[100.0, 50.0], [200.0, 50.0]]
    assert _is_normalized(poly) is False


def test_vehicle_detector_singleton():
    d1 = VehicleDetector()
    d2 = VehicleDetector()
    d3 = get_detector()
    assert d1 is d2
    assert d2 is d3


def test_parking_layout_loads_8_bays():
    slots = load_parking_layout()
    assert len(slots) == 8
    slot_ids = [s["slot_id"] for s in slots]
    assert "A1" in slot_ids
    assert "B4" in slot_ids


def test_polygon_coords_are_normalized():
    """Verify all bays in config use normalized coordinates."""
    slots = load_parking_layout()
    for slot in slots:
        for pt in slot["polygon"]:
            assert 0.0 <= pt[0] <= 1.0, f"Non-normalized x in {slot['slot_id']}: {pt}"
            assert 0.0 <= pt[1] <= 1.0, f"Non-normalized y in {slot['slot_id']}: {pt}"


def test_geometric_overlap_with_normalized_poly_and_image_shape():
    """Occupancy engine correctly scales normalized polygon and detects overlap."""
    engine = SlotOccupancyEngine(iou_threshold=0.20)

    # Normalized polygon → [0.1,0.1]-[0.5,0.5] on a 200x200 image = [20,20]-[100,100]
    slot = {
        "slot_id": "A1",
        "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
        "status": "AVAILABLE",
    }
    # Bbox overlapping the scaled polygon
    det = VehicleDetection(class_name="car", confidence=0.92, bbox=[25.0, 25.0, 90.0, 90.0])
    result = engine.compute_slot_status(slot, [det], image_shape=(200, 200))
    assert result["status"] == "OCCUPIED"
    assert result["confidence"] >= 0.80


def test_occupancy_engine_no_overlap():
    engine = SlotOccupancyEngine(iou_threshold=0.20)
    slot = {
        "slot_id": "A2",
        "polygon": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
        "status": "AVAILABLE",
    }
    # bbox far outside the polygon area (even after scaling)
    det = VehicleDetection(class_name="car", confidence=0.90, bbox=[900.0, 900.0, 990.0, 990.0])
    result = engine.compute_slot_status(slot, [det], image_shape=(480, 640))
    assert result["status"] == "AVAILABLE"


def test_point_in_polygon():
    poly = [[100.0, 50.0], [200.0, 50.0], [200.0, 150.0], [100.0, 150.0]]
    assert is_point_in_polygon((150, 100), poly) is True
    assert is_point_in_polygon((50, 50), poly) is False


def test_box_iou_perfect():
    iou = calculate_box_iou([0, 0, 10, 10], [0, 0, 10, 10])
    assert iou == 1.0


def test_box_iou_no_overlap():
    iou = calculate_box_iou([0, 0, 10, 10], [20, 20, 30, 30])
    assert iou == 0.0


# ── Background subtraction detector ─────────────────────────────────────────

def test_bg_detector_no_reference_returns_unknown():
    """Without a reference frame, each bay returns UNKNOWN."""
    import uuid
    detector = ReferenceFrameDetector(reference_path=f"missing_{uuid.uuid4()}.npy")
    slot = {
        "slot_id": "A1",
        "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
    }
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect_bay_occupancy(frame, slot, image_shape=(480, 640))
    assert result["status"] == "UNKNOWN"

def test_bg_detector_empty_lot_is_available():
    """Set reference = current frame (empty lot) → all bays AVAILABLE."""
    import uuid
    detector = ReferenceFrameDetector(reference_path=f"temp_{uuid.uuid4()}.npy")
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)  # uniform gray
    detector.set_reference(frame)

    slot = {
        "slot_id": "A1",
        "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
    }
    result = detector.detect_bay_occupancy(frame, slot, image_shape=(480, 640))
    assert result["status"] == "AVAILABLE"
    assert result["mean_diff"] < 1.0


def test_bg_detector_occupied_bay():
    """Draw a bright rectangle in the slot ROI → should be OCCUPIED."""
    import uuid
    detector = ReferenceFrameDetector(reference_path=f"temp_{uuid.uuid4()}.npy")
    reference = np.full((480, 640, 3), 60, dtype=np.uint8)
    detector.set_reference(reference)

    # Current frame has a bright region in the A1 normalized area
    current = reference.copy()
    # A1 poly [[0.1,0.1],[0.5,0.1],[0.5,0.5],[0.1,0.5]] → [64,48]-[320,240]
    cv2.rectangle(current, (70, 55), (310, 230), (255, 255, 255), -1)

    slot = {
        "slot_id": "A1",
        "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
    }
    result = detector.detect_bay_occupancy(current, slot, image_shape=(480, 640))
    assert result["status"] == "OCCUPIED"
    assert result["mean_diff"] > 25.0


# ── API endpoint tests ────────────────────────────────────────────────────────

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "detection_mode" in data
        assert "bg_calibrated" in data


def test_slots_endpoint_returns_8_bays():
    with TestClient(app) as client:
        response = client.get("/api/slots")
        assert response.status_code == 200
        data = response.json()
        assert "slots" in data
        assert len(data["slots"]) == 8


def test_image_upload_endpoint():
    with TestClient(app) as client:
        img = Image.new("RGB", (640, 480), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post("/api/detect/image", files={"file": ("test.jpg", buf, "image/jpeg")})
        assert response.status_code == 200
        data = response.json()
        assert "slots" in data
        assert "detection_mode" in data


def test_reference_frame_endpoint():
    """POST /api/reference-frame accepts an image and returns success."""
    with TestClient(app) as client:
        img = Image.new("RGB", (640, 480), color=(60, 60, 60))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post("/api/reference-frame", files={"file": ("ref.jpg", buf, "image/jpeg")})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["has_reference"] is True


def test_corrupted_image_returns_400():
    with TestClient(app) as client:
        bad = io.BytesIO(b"not_valid_image_bytes_xxxx")
        response = client.post("/api/detect/image", files={"file": ("bad.jpg", bad, "image/jpeg")})
        assert response.status_code == 400


def test_dynamic_layout_update():
    from app.services.parking_service import parking_service
    import json
    
    with TestClient(app) as client:
        # Save current layout
        old_layout = parking_service.state_manager.get_slots()
        
        custom = {
            "slots": [
                {"slot_id": "A1", "polygon": [[0.05, 0.05], [0.30, 0.05], [0.30, 0.45], [0.05, 0.45]], "status": "AVAILABLE"},
                {"slot_id": "A2", "polygon": [[0.35, 0.05], [0.60, 0.05], [0.60, 0.45], [0.35, 0.45]], "status": "AVAILABLE"},
            ]
        }
        response = client.post("/api/config/layout", json=custom)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_slots"] == 2
        
        # Restore layout
        client.post("/api/config/layout", json={"slots": old_layout})



def test_detector_annotation_runs():
    detector = get_detector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = detector.annotate_frame(frame, [], slots=[])
    assert annotated.shape == (480, 640, 3)
    b64 = detector.encode_base64_jpeg(annotated)
    assert b64.startswith("data:image/jpeg;base64,")
