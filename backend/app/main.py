from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env variables first
load_dotenv()

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.websocket import manager as ws_manager, websocket_endpoint
from app.core.config import settings
from app.core.database import create_tables
from app.services.parking_service import parking_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    # 1. Create DB tables if they don't exist
    create_tables()

    # 2. Load parking layout from config
    parking_service.initialize_demo_state()

    # 3. Wire WebSocket broadcast into state manager
    parking_service.state_manager.set_ws_callback(ws_manager.broadcast)

    yield
    # ── Shutdown (nothing needed for SQLite) ─────────────────────────────


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# Locked CORS — only allow known origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        settings.frontend_url,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.websocket("/ws")
async def ws_route(websocket: WebSocket) -> None:
    await websocket_endpoint(websocket)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}
