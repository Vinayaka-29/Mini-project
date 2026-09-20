# AI-PARK: Intelligent Miniature Parking System

An end-to-end autonomous parking management system designed for a miniature (cardboard) parking lot, featuring real-time vision-based occupancy detection, a dynamic 3D web dashboard, and smart slot allocation.

## Architecture

*   **Backend:** FastAPI (Python)
*   **Vision Engine:** Fixed-camera background subtraction (Primary) / YOLOv8 (Optional)
*   **Database:** SQLite via SQLAlchemy
*   **Real-time:** WebSocket state broadcasts
*   **Frontend:** React + Vite, Three.js (3D visualization)

## How Detection Works (Important)

This system uses a **fixed-camera background subtraction** approach as its primary detection mode, which is far more reliable for miniature/cardboard parking lots than generic COCO-pretrained models (which expect real-world street views).

1.  **Reference Frame:** The system requires an image of the *empty* parking lot to act as a baseline.
2.  **Calibration:** Parking bays are defined as normalized polygons in `configs/parking_layout.json`. You **must** use the frontend calibration tool (`/calibration_tool.html`) to trace the actual bays on a frame from your camera.
3.  **Detection:** When a new frame arrives, the system compares the Region of Interest (ROI) for each bay against the reference frame using absolute pixel difference and edge density. If thresholds are exceeded, the bay is marked `OCCUPIED`.

*Note: YOLOv8 is still available as an optional mode, but requires you to train and provide a fine-tuned model on your specific miniature cars to be effective.*

## Getting Started

### 1. Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Start the server (runs on port 8000)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
cd frontend
npm install

# Start the dev server (runs on port 5173)
npm run dev
```

### 3. Initial Calibration (Required)

Before the AI vision engine can work, you must calibrate it to your specific camera view:

1.  Open the frontend and navigate to the **Calibration Studio** (link in top right).
2.  Upload a clear image of your parking lot from its fixed camera position.
3.  Draw polygons around each parking bay (A1-A4, B1-B4) and save the resulting JSON to `backend/configs/parking_layout.json`.
4.  In the main dashboard, upload an image of the *completely empty* parking lot using the "Upload Reference" button in the warning banner. This sets the baseline for background subtraction.

## Key API Endpoints

*   `GET /api/health`: System health and calibration status.
*   `GET /api/overview`: Current occupancy statistics.
*   `GET /ws`: WebSocket endpoint for real-time state updates.
*   `POST /api/detect/image`: Upload an image to trigger detection and update bay states.
*   `POST /api/reference-frame`: Upload the empty-lot reference frame for calibration.
*   `POST /api/allocate`: Request a smart bay allocation (VIP pass).
*   `POST /api/config/detection-mode`: Switch between `background_subtraction` and `yolo`.

## Database

The system uses SQLite to persist bay states, event logs, and issued tickets across restarts. The database file is stored at `backend/data/aipark.db`.

*Note: If deploying on a service like Render's free tier, the file system is ephemeral and the database will reset on each deployment.*
