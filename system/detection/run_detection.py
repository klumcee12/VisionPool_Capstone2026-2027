"""
VisionPool Detection Script
===========================
Run alongside the Django server:
    python detection/run_detection.py

Environment variables (all optional):
    VISIONPOOL_API    Django API endpoint  (default: http://127.0.0.1:8000/api/violations/)
    CAMERA_SOURCE     Webcam index, RTSP URL, or path to a recorded video file (default: 0)
    MODEL_PATH        Path to trained YOLOv8 .pt file (default: best.pt)
    CONF_THRESHOLD    Detection confidence threshold 0-1 (default: 0.70)
    DETECTION_MODE    simulation | camera | full (default: simulation)
    LOOP_VIDEO        1 = restart a recorded video when it ends (default: 0)
    SHOW_WINDOW       0 = no preview window (default: 1)
    COOLDOWN_SEC      Seconds before the same violation type can be flagged again (default: 5)

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

VIOLATION_TYPES = ['danger_height', 'diving', 'improper_attire', 'jumping', 'missing_wristband']
CONTEXT_CLASSES = ['person']   # drawn on screen for context, NEVER posted as a violation
VIOLATION_LABELS = {
    'danger_height':    'Height-to-depth risk (staff review)',
    'diving':           'Diving in restricted zone',
    'improper_attire':  'Improper attire',
    'jumping':          'Prohibited jumping',
    'missing_wristband': 'Missing wristband',
}

# Optional per-class confidence thresholds. Classes not listed use CONF_THRESHOLD.
# Example: {'missing_wristband': 0.60, 'improper_attire': 0.75}
PER_CLASS_CONF: dict = {}

COOLDOWN_SEC = float(os.environ.get('COOLDOWN_SEC', '5'))

# Recorded-video testing:  CAMERA_SOURCE=path/to/video.mp4
# LOOP_VIDEO=1 restarts the video when it ends; SHOW_WINDOW=0 disables the preview window.
IS_FILE    = isinstance(CAMERA_SOURCE, str) and not CAMERA_SOURCE.lower().startswith(('rtsp://', 'http://', 'https://'))
LOOP_VIDEO = os.environ.get('LOOP_VIDEO', '0') == '1'
SHOW_WINDOW = os.environ.get('SHOW_WINDOW', '1') == '1'



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

def _fatal(msg: str):
    print(f"[ERROR] {msg}")
    print("[ERROR] Full mode will not fall back to simulation, so fake violations never reach the dashboard.")
    sys.exit(1)


def run_full(model_path: str):
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        _fatal(f"Missing dependency: {exc}")

    if not Path(model_path).exists():
        _fatal(f"Model not found at '{model_path}'.")

    if IS_FILE and not Path(CAMERA_SOURCE).exists():
        _fatal(f"Video file not found: '{CAMERA_SOURCE}'.")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        _fatal(f"Camera/video source '{CAMERA_SOURCE}' could not be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    base_conf = min([CONF_THRESHOLD] + list(PER_CLASS_CONF.values()))

    src_kind = 'video file' if IS_FILE else 'live camera'
    print(f"[INFO] Mode: full  |  Model={model_path}  |  Source={CAMERA_SOURCE} ({src_kind}, {fps:.0f} fps)  |  API: {API_URL}")
    print("[INFO] Press Q in the OpenCV window to stop.\n")

    cooldown: dict = {}
    frame_idx = 0
    wall_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            if IS_FILE:
                if LOOP_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx, cooldown, wall_start = 0, {}, time.time()
                    continue
                print("[INFO] End of video.")
                break
            time.sleep(0.3)   # live stream hiccup: retry
            continue

        # Clock used for the cooldown: video time for files, wall time for live cameras.
        now = (frame_idx / fps) if IS_FILE else time.time()
        frame_idx += 1

        results = model(frame, conf=base_conf, verbose=False)

        flags = []
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if label in CONTEXT_CLASSES:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
                    continue
                if label not in VIOLATION_TYPES:
                    continue
                if conf < PER_CLASS_CONF.get(label, CONF_THRESHOLD):
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{VIOLATION_LABELS[label]} {conf:.0%}",
                            (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Cooldown is per violation type (known limitation: not per person).
                if now - cooldown.get(label, -1e9) < COOLDOWN_SEC:
                    continue
                cooldown[label] = now
                flags.append((label, conf))

        # Save the annotated frame once per new flag, then post it.
        for label, conf in flags:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            shot = SCREENSHOT_DIR / f"{label}_{ts}.jpg"
            cv2.imwrite(str(shot), frame)
            post_violation(label, conf, str(shot))

        if SHOW_WINDOW:
            cv2.imshow('VisionPool — Detection Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Pace recorded video to its real speed (only if inference is faster than real time).
        if IS_FILE:
            ahead = (wall_start + frame_idx / fps) - time.time()
            if ahead > 0:
                time.sleep(ahead)

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