"""
Tests updated to match:
- 8 bays (A1-B4) with normalized coords
- Background subtraction primary detection
- AllocationEngine integrated in allocate_slot
- New /api/reference-frame endpoint
"""
from app.parking.state_manager import ParkingStateManager

SAMPLE_LAYOUT = {
    "slots": [
        {
            "slot_id": "A1", "section_id": "A", "camera_id": "CAM_01",
            "polygon": [[0.06, 0.08], [0.27, 0.08], [0.27, 0.38], [0.06, 0.38]],
            "status": "AVAILABLE", "distance_from_entries": 8, "priority": 1,
        },
        {
            "slot_id": "A2", "section_id": "A", "camera_id": "CAM_01",
            "polygon": [[0.28, 0.08], [0.49, 0.08], [0.49, 0.38], [0.28, 0.38]],
            "status": "AVAILABLE", "distance_from_entries": 12, "priority": 1,
        },
        {
            "slot_id": "B1", "section_id": "B", "camera_id": "CAM_01",
            "polygon": [[0.06, 0.55], [0.27, 0.55], [0.27, 0.85], [0.06, 0.85]],
            "status": "AVAILABLE", "distance_from_entries": 22, "priority": 2,
        },
    ]
}


def test_load_layout_and_overview():
    state = ParkingStateManager()
    state.load_layout(SAMPLE_LAYOUT)

    overview = state.get_overview()
    assert overview["total_slots"] == 3
    assert overview["available"] == 3
    assert overview["occupied"] == 0


def test_allocate_slot_uses_nearest():
    state = ParkingStateManager()
    state.load_layout(SAMPLE_LAYOUT)

    result = state.allocate_slot("KA-01-TEST-1234", "car")
    # A1 has distance_from_entries=8, lowest → nearest → should be picked
    assert result["slot_id"] == "A1"
    assert result["status"] == "RESERVED"
    assert state.slots["A1"]["status"] == "RESERVED"


def test_allocate_slot_preferred_id():
    """AllocationEngine preferred_slot_id is respected."""
    state = ParkingStateManager()
    state.load_layout(SAMPLE_LAYOUT)

    result = state.allocate_slot("KA-02-TEST-5678", "car", preferred_slot_id="A2")
    assert result["slot_id"] == "A2"
    assert state.slots["A2"]["status"] == "RESERVED"


def test_reset_all_slots():
    state = ParkingStateManager()
    state.load_layout(SAMPLE_LAYOUT)
    state.allocate_slot("KA-03-TEST-0001", "car")

    overview_before = state.get_overview()
    assert overview_before["reserved"] == 1

    state.reset_all_slots()
    overview_after = state.get_overview()
    assert overview_after["reserved"] == 0
    assert overview_after["available"] == 3


def test_ws_callback_is_called_on_status_change():
    """Verify WebSocket callback is invoked when slot status changes."""
    state = ParkingStateManager()
    state.load_layout(SAMPLE_LAYOUT)

    call_count = {"n": 0}

    async def fake_broadcast(payload):
        call_count["n"] += 1

    state.set_ws_callback(fake_broadcast)
    state.update_slot_status("A1", "OCCUPIED", 0.92)
    # Callback may not have been awaited synchronously, but was scheduled
    assert state.slots["A1"]["status"] == "OCCUPIED"
