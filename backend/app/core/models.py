from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Bay(Base):
    __tablename__ = "bays"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bay_id = Column(String(32), unique=True, nullable=False, index=True)
    section_id = Column(String(8), nullable=False, default="A")
    camera_id = Column(String(32), nullable=False, default="CAM_01")
    polygon_json = Column(Text, nullable=True)  # JSON-encoded normalized polygon
    status = Column(String(16), nullable=False, default="AVAILABLE")
    confidence = Column(Float, nullable=False, default=0.98)
    priority = Column(Integer, nullable=False, default=1)
    distance_from_entries = Column(Float, nullable=False, default=0.0)
    slot_type = Column(String(32), nullable=False, default="STANDARD")
    vehicle_id = Column(String(64), nullable=True)
    occupied_since = Column(DateTime(timezone=True), nullable=True)
    reserved_since = Column(DateTime(timezone=True), nullable=True)
    last_updated = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    events = relationship("Event", back_populates="bay", lazy="dynamic")
    tickets = relationship("Ticket", back_populates="bay", lazy="dynamic")

    @property
    def polygon(self) -> list:
        if self.polygon_json:
            try:
                return json.loads(self.polygon_json)
            except Exception:
                return []
        return []

    @polygon.setter
    def polygon(self, value: list) -> None:
        self.polygon_json = json.dumps(value) if value else None

    def to_dict(self) -> dict:
        return {
            "slot_id": self.bay_id,
            "section_id": self.section_id,
            "camera_id": self.camera_id,
            "polygon": self.polygon,
            "status": self.status,
            "confidence": self.confidence,
            "priority": self.priority,
            "distance_from_entries": self.distance_from_entries,
            "type": self.slot_type,
            "vehicle_id": self.vehicle_id,
            "occupied_since": self.occupied_since.isoformat() if self.occupied_since else None,
            "reserved_since": self.reserved_since.isoformat() if self.reserved_since else None,
        }


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(32), nullable=False, index=True)
    bay_id = Column(String(32), ForeignKey("bays.bay_id"), nullable=False)
    section_id = Column(String(8), nullable=True)
    event_type = Column(String(64), nullable=False)
    previous_status = Column(String(16), nullable=True)
    status = Column(String(16), nullable=False)
    confidence = Column(Float, nullable=True)
    vehicle_id = Column(String(64), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=_now, nullable=False)

    bay = relationship("Bay", back_populates="events")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "slot_id": self.bay_id,
            "section_id": self.section_id,
            "event_type": self.event_type,
            "previous_status": self.previous_status,
            "status": self.status,
            "confidence": self.confidence,
            "vehicle_id": self.vehicle_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(32), unique=True, nullable=False, index=True)
    plate = Column(String(64), nullable=False)
    vehicle_type = Column(String(32), nullable=False, default="car")
    bay_id = Column(String(32), ForeignKey("bays.bay_id"), nullable=False)
    section_id = Column(String(8), nullable=True)
    issued_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(16), nullable=False, default="ACTIVE")  # ACTIVE / EXPIRED / CANCELLED
    qr_code_data = Column(String(128), nullable=True)

    bay = relationship("Bay", back_populates="tickets")

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "vehicle_id": self.plate,
            "vehicle_type": self.vehicle_type,
            "slot_id": self.bay_id,
            "section_id": self.section_id,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "status": self.status,
            "qr_code_data": self.qr_code_data,
        }
