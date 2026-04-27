"""
================================================================================
app.py - Web Application Server
================================================================================
Serves a real-time emotion detection web app using FastAPI + WebSockets.
Run with:  python app.py
Then open: http://localhost:8000
================================================================================
"""

import os
import sys
import json
import base64
import asyncio
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# ──────────────────────────────────────────────────────────────
# Suppress TensorFlow noise
# ──────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from model_structure import build_emotion_model
from processing import FaceProcessor

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, 'emotion_model.h5')

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ══════════════════════════════════════════════════════════════
# GLOBAL MODEL & PROCESSOR
# ══════════════════════════════════════════════════════════════
model = None
processor = None

logger = logging.getLogger("emotion_app")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app):
    """Load CNN model and face processor on server start."""
    global model, processor

    logger.info("")
    logger.info("=" * 60)
    logger.info("  EMOTION DETECTION 2.0 -- Web Server")
    logger.info("=" * 60)

    logger.info("[INFO] Building CNN model...")
    model = build_emotion_model()

    if os.path.exists(MODEL_WEIGHTS_PATH):
        logger.info(f"[INFO] Loading weights from: {MODEL_WEIGHTS_PATH}")
        model.load_weights(MODEL_WEIGHTS_PATH)
        logger.info("[INFO] Weights loaded successfully!")
    else:
        logger.warning("[WARN] No weights found -- predictions will be random.")

    processor = FaceProcessor()
    logger.info("[INFO] Server ready! Open http://localhost:8000")
    logger.info("")
    yield


# ══════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="Emotion Detection 2.0", lifespan=lifespan)

# Serve static files (CSS, JS)
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time frame processing."""
    await websocket.accept()

    try:
        while True:
            # Receive base64-encoded JPEG frame from browser
            data = await websocket.receive_text()

            # Decode the image
            try:
                header, encoded = data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"faces": []})
                continue

            if frame is None:
                await websocket.send_json({"faces": []})
                continue

            # Detect faces
            faces = processor.detect_faces(frame)
            frame_h, frame_w = frame.shape[:2]

            results = []
            for (x, y, w, h) in faces:
                # Preprocess and predict
                processed = processor.preprocess_face(frame, x, y, w, h)
                predictions = model.predict(processed, verbose=0)
                probs = predictions[0].tolist()
                emotion_idx = int(np.argmax(probs))

                results.append({
                    "x": int(x), "y": int(y),
                    "w": int(w), "h": int(h),
                    "emotion": EMOTION_LABELS[emotion_idx],
                    "confidence": float(probs[emotion_idx]),
                    "probabilities": {
                        EMOTION_LABELS[i]: round(float(probs[i]), 4)
                        for i in range(7)
                    }
                })

            await websocket.send_json({
                "faces": results,
                "frameWidth": frame_w,
                "frameHeight": frame_h
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"[WS ERROR] {e}")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
