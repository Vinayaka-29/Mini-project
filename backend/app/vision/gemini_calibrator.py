import os
import json
import re
import base64
import time
import requests

CALIBRATION_PROMPT = """Analyze this parking lot image and return a JSON array of parking bay locations.

Each bay must be an object with:
- "id": string like "A1", "A2", "B1", "B2" (row letter + column number, sorted top-to-bottom then left-to-right)
- "box": array of 4 normalized floats [x_min, y_min, x_max, y_max] where 0.0-1.0 represents image width/height

Example: [{"id": "A1", "box": [0.1, 0.2, 0.3, 0.4]}, {"id": "A2", "box": [0.35, 0.2, 0.55, 0.4]}]

Rules:
- Coordinates MUST be normalized (0.0 to 1.0), NOT pixel values
- Only include bays with visible parking lines or clear boundaries
- Sort top-to-bottom, left-to-right
- Return empty array [] if no bays found
"""

def calibrate_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> list[dict]:
    """Calls Gemini with the image, returns a list of {id, box} dicts.
    Raises ValueError if the response can't be parsed as valid bay data."""

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_model = "gemini-3.5-flash"  # More stable than 3.6
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_api_key}"

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [{
            "parts": [
                {"text": CALIBRATION_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": b64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"
        }
    }

    # Retry logic for 503 errors and timeouts
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(gemini_url, json=payload, timeout=60)  # Increased timeout
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise ValueError("Gemini API timed out after 3 retries. Try a smaller image.")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            raise ValueError(f"Gemini API error: {e.response.status_code} {e.response.reason}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error calling Gemini: {str(e)}")
    else:
        raise ValueError("Gemini API unavailable after 3 retries")

    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response shape: {data}") from e

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    # Clean up accidental double commas or trailing commas
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    try:
        bays = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini did not return valid JSON: {raw_text}") from e

    if not isinstance(bays, list):
        raise ValueError(f"Expected a JSON array, got: {type(bays)}")

    for bay in bays:
        if not isinstance(bay, dict) or "id" not in bay or "box" not in bay:
            raise ValueError(f"Malformed bay entry: {bay}")
        box = bay["box"]
        if not (isinstance(box, list) and len(box) == 4):
            raise ValueError(f"Bay box must be [x_min,y_min,x_max,y_max]: {bay}")
        x_min, y_min, x_max, y_max = box
        if not all(0.0 <= v <= 1.0 for v in box):
            raise ValueError(f"Bay coordinates must be normalized 0-1: {bay}")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"Bay box has zero/negative area: {bay}")

    return bays
