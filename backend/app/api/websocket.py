from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Total: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Total: %d", len(self.active_connections))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        """Broadcast a JSON payload to all connected clients."""
        message = json.dumps(payload)
        dead: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead.append(connection)
        for ws in dead:
            if ws in self.active_connections:
                self.active_connections.remove(ws)


# Module-level singleton
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle a WebSocket connection, keep alive until disconnect."""
    from app.services.parking_service import parking_service

    await manager.connect(websocket)

    # Send current state immediately on connect
    try:
        await websocket.send_text(json.dumps({
            "type": "STATE_UPDATE",
            "overview": parking_service.get_overview(),
            "slots": parking_service.get_slots(),
            "events": parking_service.get_recent_events(),
        }))
    except Exception:
        pass

    try:
        while True:
            # Keep connection alive — client sends pings, we echo back
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
