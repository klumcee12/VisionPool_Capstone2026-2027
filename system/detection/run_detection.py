"""
VisionPool Detection Script
===========================
Run alongside the Django server:
    python detection/run_detection.py

Environment variables (all optional):
    VISIONPOOL_API    Django API endpoint  (default: http://127.0.0.1:8000/api/violations/)
    CAMERA_SOURCE     Webcam index or RTSP URL (default: 0)
    MODEL_PATH        Path to trained YOLOv8 .pt file (default: best.pt)
    CONF_THRESHOLD    Detection confidence threshold 0-1 (default: 0.70)
    DETECTION_MODE    simulation | camera | full (default: simulation)

Modes:
    simulation  No camera or model needed. Generates random violations and posts
                them to Django. Best for testing the dashboard end-to-end.
    camera      Opens the webcam and saves screenshots, but uses simulated detections.
                Useful for testing the video pipeline before the custom model is ready.
    full        Full pipeline: real webcam + trained YOLOv8 model. Switch to this
                once your custom model (best.pt) is trained in Ultralytics.
"""

import os
import sys
import time
import random
import requests
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

API_URL           = os.environ.get('VISIONPOOL_API',    'http://127.0.0.1:8000/api/violations/')
CAMERA_SOURCE_RAW = os.environ.get('CAMERA_SOURCE',    '0')
MODEL_PATH        = os.environ.get('MODEL_PATH',        'best.pt')
CONF_THRESHOLD    = float(os.environ.get('CONF_THRESHOLD', '0.70'))
DETECTION_MODE    = os.environ.get('DETECTION_MODE',   'simulation')

# Try to parse as int (webcam index); fall back to string (RTSP URL)
try:
    CAMERA_SOURCE = int(CAMERA_SOURCE_RAW)
except ValueError:
    CAMERA_SOURCE = CAMERA_SOURCE_RAW

SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / 'screenshots'
SCREENSHOT_DIR.mkdir(exist_ok=True)

VIOLATION_TYPES = ['running', 'diving', 'attire', 'object', 'wristband']
VIOLATION_LABELS = {
    'running':   'Running on deck',
    'diving':    'Diving in restricted zone',
    'attire':    'Improper attire',
    'object':    'Prohibited object',
    'wristband': 'Missing wristband',
}


def post_violation(vtype: str, confidence: float, screenshot_path: str = '', camera_id: str = 'CAM-01'):
    payload = {
        'type':            vtype,
        'confidence':      round(confidence, 3),
        'screenshot_path': screenshot_path,
        'camera_id':       camera_id,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=3)
        r.raise_for_status()
        inc_id = r.json().get('id', '?')
        print(f"[{datetime.now():%H:%M:%S}]  FLAGGED  {VIOLATION_LABELS[vtype]:<30}  conf={confidence:.0%}  db_id={inc_id}")
    except requests.exceptions.ConnectionError:
        print(f"[{datetime.now():%H:%M:%S}]  ERROR    Cannot reach Django at {API_URL}  — is the server running?")
    except Exception as exc:
        print(f"[{datetime.now():%H:%M:%S}]  ERROR    {exc}")


# ---------------------------------------------------------------------------
# Mode 1 – Headless simulation (no camera, no model)
# ---------------------------------------------------------------------------

def run_simulation():
    print(f"[INFO] Mode: simulation  |  API: {API_URL}")
    print("[INFO] Generating random violations. Press Ctrl+C to stop.\n")
    cooldown: dict = {}
    while True:
        time.sleep(random.uniform(4, 12))
        vtype = random.choice(VIOLATION_TYPES)
        now = time.time()
        if now - cooldown.get(vtype, 0) < 8:
            continue
        cooldown[vtype] = now
        conf = round(random.uniform(0.72, 0.97), 2)
        post_violation(vtype, conf)


# ---------------------------------------------------------------------------
# Mode 2 – Camera + simulated detections (no model)
# ---------------------------------------------------------------------------

def run_camera_simulation():
    try:
        import cv2
    except ImportError:
        print("[WARNING] opencv-python not installed. Falling back to headless simulation.")
        run_simulation()
        return

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[WARNING] Camera source '{CAMERA_SOURCE}' not available. Falling back to headless simulation.")
        run_simulation()
        return

    print(f"[INFO] Mode: camera simulation  |  Camera={CAMERA_SOURCE}  |  API: {API_URL}")
    print("[INFO] Press Q in the OpenCV window to stop.\n")

    frame_no = 0
    cooldown: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame read failed — retrying...")
            time.sleep(0.3)
            continue

        frame_no += 1
        cv2.putText(frame, 'VisionPool  (simulation mode)', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 90), 1)

        # Simulate a detection roughly once every 5 seconds at 30 fps (~150 frames)
        if frame_no % 150 == 0 or (frame_no % 30 == 0 and random.random() < 0.04):
            vtype = random.choice(VIOLATION_TYPES)
            now = time.time()
            if now - cooldown.get(vtype, 0) >= 6:
                cooldown[vtype] = now
                conf = round(random.uniform(0.72, 0.97), 2)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                shot = SCREENSHOT_DIR / f"{vtype}_{ts}.jpg"
                cv2.imwrite(str(shot), frame)
                # Draw a placeholder bounding box on the frame for visual feedback
                h_f, w_f = frame.shape[:2]
                x1, y1 = random.randint(30, w_f//2), random.randint(30, h_f//2)
                x2, y2 = x1 + random.randint(60, 120), y1 + random.randint(80, 150)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{VIOLATION_LABELS[vtype]} {conf:.0%}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                post_violation(vtype, conf, str(shot))

        cv2.imshow('VisionPool — Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Mode 3 – Full pipeline: trained YOLOv8 model + camera
# ---------------------------------------------------------------------------

def run_full(model_path: str):
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[WARNING] Missing dependency: {exc}. Falling back to camera simulation.")
        run_camera_simulation()
        return

    if not Path(model_path).exists():
        print(f"[WARNING] Model not found at '{model_path}'.")
        print("[WARNING] Train your YOLOv8 model first and place the weights at that path.")
        print("[WARNING] Falling back to camera simulation.\n")
        run_camera_simulation()
        return

    model = YOLO(model_path)
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[WARNING] Camera source '{CAMERA_SOURCE}' not available. Falling back to headless simulation.")
        run_simulation()
        return

    print(f"[INFO] Mode: full  |  Model={model_path}  |  Camera={CAMERA_SOURCE}  |  API: {API_URL}")
    print("[INFO] Press Q in the OpenCV window to stop.\n")

    cooldown: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.3)
            continue

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = model.names[cls_id]

                if label not in VIOLATION_TYPES:
                    continue

                now = time.time()
                if now - cooldown.get(label, 0) < 5:
                    continue
                cooldown[label] = now

                # Annotate frame and save screenshot
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{VIOLATION_LABELS[label]} {conf:.0%}",
                            (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
                shot = SCREENSHOT_DIR / f"{label}_{ts}.jpg"
                cv2.imwrite(str(shot), frame)
                post_violation(label, conf, str(shot))

        cv2.imshow('VisionPool — Detection Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    mode = DETECTION_MODE.lower()
    print('=' * 55)
    print('  VisionPool Detection Script')
    print('=' * 55)

    if mode == 'full':
        run_full(MODEL_PATH)
    elif mode == 'camera':
        run_camera_simulation()
    else:
        run_simulation()
