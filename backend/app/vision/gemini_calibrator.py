import os
import json
import re
import base64
import time
import requests

CALIBRATION_PROMPT = """You are a computer vision system that finds parking
bay boundaries in an image of a parking lot.

Look at this image and identify every distinct parking bay (a marked or
implied rectangular space where one vehicle parks — whether empty or
occupied).

Return ONLY a JSON array, nothing else — no markdown fences, no prose,
no explanation before or after. Each element must have this exact shape:

{"id": "A1", "box": [x_min, y_min, x_max, y_max]}

Rules:
- Coordinates are normalized floats between 0.0 and 1.0, relative to the
  image width (x) and height (y).
- Sort bays by row (top to bottom), then by column (left to right) within
  a row. Label rows A, B, C... and number left to right: A1, A2, A3...
  then B1, B2, B3... and so on.
- Only return bays you can actually see evidence of (painted lines, curb
  markings, or a clearly implied parking space). Do not force any
  specific count. Do not invent bays over grass, paths, buildings, or
  open road with no parking markings.
- If you cannot identify any parking bays at all, return an empty array: []
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
            "maxOutputTokens": 8192
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

    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response shape: {data}") from e

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

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
