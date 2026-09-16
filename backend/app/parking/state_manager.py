from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from app.vision.occupancy import SlotOccupancyEngine


class ParkingStateManager:
    def __init__(self) -> None:
        self.slots: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.allocations: dict[str, str] = {}
        self.occupancy_engine = SlotOccupancyEngine(iou_threshold=0.20)
        self.last_analysis_time: str | None = None
        self.total_detections: int = 0

    def load_layout(self, config: dict[str, Any] | list[dict[str, Any]]) -> None:
        self.slots = {}
        slot_list = config if isinstance(config, list) else config.get("slots", [])
        for raw_slot in slot_list:
            slot_id = str(raw_slot.get("slot_id") or raw_slot.get("id"))
            slot = {
                "slot_id": slot_id,
                "section_id": raw_slot.get("section_id", "A"),
                "camera_id": raw_slot.get("camera_id", "CAM_01"),
                "polygon": raw_slot.get("polygon", []),
                "status": raw_slot.get("status", "AVAILABLE"),
                "type": raw_slot.get("type", "STANDARD"),
                "priority": raw_slot.get("priority", 1),
                "distance_from_entries": raw_slot.get("distance_from_entries", 0),
                "confidence": float(raw_slot.get("confidence", 0.98)),
                "occupied_since": None,
                "reserved_since": None,
                "vehicle_id": None,
            }
            self.slots[slot["slot_id"]] = slot

    def update_layout(self, layout: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        """Dynamically update parking spot polygons in memory."""
        self.load_layout(layout)
        return {
            "status": "success",
            "message": "Parking layout updated successfully",
            "total_slots": len(self.slots),
            "slots": self.get_slots(),
        }


    def expire_reservations(self, max_minutes: int = 15) -> None:
        """Expire old reservations if not claimed within timeout."""
        now = datetime.now(timezone.utc)
        for slot in self.slots.values():
            if slot.get("status") == "RESERVED" and slot.get("reserved_since"):
                try:
                    res_time = datetime.fromisoformat(slot["reserved_since"])
                    if (now - res_time).total_seconds() > (max_minutes * 60):
                        slot["status"] = "AVAILABLE"
                        slot["vehicle_id"] = None
                        slot["reserved_since"] = None
                except Exception:
                    pass

    def update_slot_status(self, slot_id: str, status: str, confidence: float = 0.95, vehicle_id: str | None = None) -> None:
        if slot_id not in self.slots:
            return
        
        old_status = self.slots[slot_id]["status"]
        if old_status == status and self.slots[slot_id].get("vehicle_id") == vehicle_id:
            return

        self.slots[slot_id]["status"] = status
        self.slots[slot_id]["confidence"] = confidence
        self.slots[slot_id]["vehicle_id"] = vehicle_id
        now_str = datetime.now(timezone.utc).isoformat()

        if status == "OCCUPIED":
            self.slots[slot_id]["occupied_since"] = now_str
            self.slots[slot_id]["reserved_since"] = None
        elif status == "RESERVED":
            self.slots[slot_id]["reserved_since"] = now_str
        elif status == "AVAILABLE":
            self.slots[slot_id]["occupied_since"] = None
            self.slots[slot_id]["reserved_since"] = None

        self.events.append({
            "event_id": f"EVT-{len(self.events) + 1:04d}",
            "slot_id": slot_id,
            "section_id": self.slots[slot_id].get("section_id", "A"),
            "event_type": "SLOT_STATUS_CHANGED" if not vehicle_id else "VEHICLE_ASSIGNED",
            "timestamp": now_str,
            "previous_status": old_status,
            "status": status,
            "confidence": confidence,
            "vehicle_id": vehicle_id,
        })

    def process_vision_detections(self, detections: list[Any], image_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        """Apply vehicle detections to compute slot occupancy."""
        self.last_analysis_time = datetime.now(timezone.utc).isoformat()
        self.total_detections = len(detections)

        updated_slots = []
        for slot in self.slots.values():
            res = self.occupancy_engine.compute_slot_status(slot, detections, image_shape)
            new_status = res["status"]  # Pure visual ground truth: OCCUPIED or AVAILABLE
            self.update_slot_status(slot["slot_id"], new_status, res["confidence"])
            updated_slots.append(self.slots[slot["slot_id"]])

        return self.get_overview()

    def simulate_random_change(self) -> dict[str, Any]:
        """Simulate a vehicle arriving or leaving a random slot for dynamic demo."""
        if not self.slots:
            return self.get_overview()

        slot_id = random.choice(list(self.slots.keys()))
        current_status = self.slots[slot_id]["status"]
        new_status = "OCCUPIED" if current_status == "AVAILABLE" else "AVAILABLE"
        vehicle_plate = f"KA-0{random.randint(1,9)}-{random.choice(['AI','CY','GT','MK'])}-{random.randint(1000, 9999)}" if new_status == "OCCUPIED" else None
        
        self.update_slot_status(slot_id, new_status, round(random.uniform(0.92, 0.99), 2), vehicle_plate)
        return self.get_overview()

    def reset_all_slots(self) -> dict[str, Any]:
        """Reset all slots to available."""
        for slot_id in self.slots:
            self.update_slot_status(slot_id, "AVAILABLE", 0.99, None)
        return self.get_overview()

    def get_overview(self) -> dict[str, Any]:
        self.expire_reservations()
        total = len(self.slots)
        available = sum(1 for slot in self.slots.values() if slot["status"] == "AVAILABLE")
        occupied = sum(1 for slot in self.slots.values() if slot["status"] == "OCCUPIED")
        reserved = sum(1 for slot in self.slots.values() if slot["status"] == "RESERVED")
        unknown = sum(1 for slot in self.slots.values() if slot["status"] == "UNKNOWN")
        occupancy = round((occupied / total) * 100, 1) if total else 0.0
        return {
            "total_slots": total,
            "available": available,
            "occupied": occupied,
            "reserved": reserved,
            "unknown": unknown,
            "occupancy_pct": occupancy,
            "last_analysis_time": self.last_analysis_time,
            "active_detections": self.total_detections,
        }

    def get_slots(self) -> list[dict[str, Any]]:
        self.expire_reservations()
        return list(self.slots.values())

    def get_recent_events(self) -> list[dict[str, Any]]:
        return self.events[-15:]

    def allocate_slot(self, vehicle_id: str, vehicle_type: str = "car") -> dict[str, Any]:
        self.expire_reservations()
        candidate = None
        for slot in self.slots.values():
            if slot["status"] == "AVAILABLE":
                if candidate is None or slot["distance_from_entries"] < candidate["distance_from_entries"]:
                    candidate = slot
        if candidate is None:
            raise ValueError("Parking Full! No available slots currently.")

        slot_id = candidate["slot_id"]
        self.allocations[vehicle_id] = slot_id
        self.update_slot_status(slot_id, "RESERVED", 0.95, vehicle_id)
        
        return {
            "ticket_id": f"TKT-{random.randint(10000, 99999)}",
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "slot_id": slot_id,
            "section_id": candidate["section_id"],
            "status": "RESERVED",
            "distance": candidate["distance_from_entries"],
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "qr_code_data": f"AI-PARK:PASS:{vehicle_id}:{slot_id}",
        }

    def get_allocations(self) -> dict[str, str]:
        return self.allocations

    def get_vehicles(self) -> list[dict[str, Any]]:
        vehicles = []
        for slot in self.slots.values():
            if slot.get("vehicle_id"):
                vehicles.append({
                    "vehicle_id": slot["vehicle_id"],
                    "slot_id": slot["slot_id"],
                    "status": slot["status"],
                    "since": slot.get("occupied_since") or slot.get("reserved_since"),
                })
        return vehicles

