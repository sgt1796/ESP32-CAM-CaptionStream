"""
Flask ESP32‑CAM demo with **YOLO‑v10 object tracking** (stream=True) + captioning
==============================================================================
Uses Ultralytics’ tracker in **stream mode** so tracking state persists across
calls and gives smoother, non‑flashy boxes.

Key options
-----------
* `USE_YOLO_TRACK`   – turn tracking on/off.
* `TRACKER_CFG`      – choose tracker YAML (`bytetrack.yaml`, `botsort.yaml`).
* `CAPTION_INTERVAL` – seconds between caption requests.

Install & run
-------------
```bash
pip install flask aiohttp requests opencv-python pillow numpy ultralytics
python app_yolo.py   # browse to http://<HOST>:5000/
```
"""
from __future__ import annotations

import asyncio
import base64
import threading
import time
from typing import Optional

import aiohttp
import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ESP32_STREAM_URL   = "http://192.168.0.199:81/stream"  # camera MJPEG endpoint
CAPTION_API_URL    = ""               # empty ⇒ use local LLM wrapper
CAPTION_MODEL      = "openai"         # "openai" | "gemini"
CAPTION_INTERVAL   = 3                # seconds between caption calls

YOLO_MODEL_PATH    = "yolo11n.pt"     # CPU‑friendly nano model
YOLO_CONF_THRESH   = 0.35             # detection confidence threshold
USE_YOLO_TRACK     = True             # enable tracking overlay
TRACKER_CFG        = "bytetrack.yaml" # or "botsort.yaml"

CAPTION_AREA_H     = 120              # pixel height of caption strip
VERIFY_SSL         = False            # accept self‑signed certs on API URL

# ─── HEAVY OBJECTS ─────────────────────────────────────────────────────────────
print("[INFO] Loading YOLO model …")
yolo = YOLO(YOLO_MODEL_PATH)

llm = None
if not CAPTION_API_URL:
    from llm import LLM  # user‑supplied class
    llm = LLM(model=CAPTION_MODEL)

# ─── GLOBAL STATE ──────────────────────────────────────────────────────────────
latest_jpeg: Optional[bytes] = None
latest_caption: str = ""
stop_event = threading.Event()

# ─── BACKGROUND: GRABBER ───────────────────────────────────────────────────────

def frame_reader() -> None:
    """Continuously capture JPEG frames from ESP32‑CAM."""
    global latest_jpeg
    while not stop_event.is_set():
        cap = cv2.VideoCapture(ESP32_STREAM_URL)
        if not cap.isOpened():
            print("[WARN] Cannot open stream. Retry in 2 s …")
            time.sleep(2)
            continue
        print("[INFO] Connected to ESP32 stream.")
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed — reconnecting …")
                break
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                latest_jpeg = buf.tobytes()
        cap.release()

# ─── BACKGROUND: CAPTION WORKER ────────────────────────────────────────────────
async def caption_worker() -> None:
    global latest_caption
    if CAPTION_API_URL:
        connector = aiohttp.TCPConnector(ssl=VERIFY_SSL is True)
        async with aiohttp.ClientSession(connector=connector) as session:
            while not stop_event.is_set():
                await asyncio.sleep(CAPTION_INTERVAL)
                if not latest_jpeg:
                    continue
                data = aiohttp.FormData()
                data.add_field("model", CAPTION_MODEL)
                data.add_field("image_file", latest_jpeg, filename="frame.jpg", content_type="image/jpeg")
                try:
                    async with session.post(CAPTION_API_URL, data=data) as resp:
                        latest_caption = (await resp.text()).strip() if resp.status == 200 else ""
                except Exception as e:
                    print(f"[Caption API] {e}")
    else:
        while not stop_event.is_set():
            await asyncio.sleep(CAPTION_INTERVAL)
            if not latest_jpeg:
                continue
            try:
                b64 = base64.b64encode(latest_jpeg).decode()
                latest_caption = llm.image_caption(imgbase64=b64).strip()
            except Exception as e:
                print(f"[Caption LLM] {e}")

def start_caption_thread():
    asyncio.run(caption_worker())

# ─── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!doctype html><title>ESP32‑CAM • YOLO‑Track • Caption</title>
<style>body{background:#111;color:#eee;font-family:Arial;text-align:center}
img{max-width:100%;border:4px solid #444;border-radius:8px}
.caption{margin-top:6px;font:14px/1.4 monospace;white-space:pre-wrap}</style>
<h1>ESP32‑CAM  •  YOLO‑tracking  •  Caption</h1>
<img src="{{ url_for('video_feed') }}"><div class=caption id=cap></div>
<script>async function poll(){const r=await fetch('/caption',{cache:'no-store'});
if(r.ok) document.getElementById('cap').textContent=await r.text();}
setInterval(poll,{{interval}});poll();</script>"""

@app.route('/')
def index():
    return render_template_string(HTML, interval=CAPTION_INTERVAL*1000)

@app.route('/caption')
def caption():
    return latest_caption or "", 200, {"Cache-Control":"no-store"}

# ─── MJPEG GENERATOR ──────────────────────────────────────────────────────────

def mjpeg_generator():
    font, scale, thick, col = cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1, (255,255,255)
    while True:
        if latest_jpeg is None:
            time.sleep(0.01)
            continue
        arr = np.frombuffer(latest_jpeg, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # ─── Detection / Tracking ────────────────────────────────────────
        try:
            if USE_YOLO_TRACK:
                stream_gen = yolo.track(frame, stream=True, persist=True, conf=YOLO_CONF_THRESH,
                                         tracker=TRACKER_CFG, verbose=False)
                res = next(stream_gen)  # yields exactly one result for our single frame
            else:
                res = yolo(frame, conf=YOLO_CONF_THRESH, verbose=False)[0]
            frame = res.plot()
        except Exception as e:
            print(f"[YOLO] {e}")

        # ─── Caption strip ───────────────────────────────────────────────
        if latest_caption:
            h, w = frame.shape[:2]
            strip = np.zeros((CAPTION_AREA_H, w, 3), np.uint8)
            words, line, lines = latest_caption.split(), "", []
            for word in words:
                if len(line)+len(word) < 60:
                    line += word + " "
                else:
                    lines.append(line); line = word + " "
            lines.append(line)
            for i, ln in enumerate(lines[:5]):
                cv2.putText(strip, ln.strip(), (10, 20+i*20), font, scale, col, thick, cv2.LINE_AA)
            frame = np.vstack((frame, strip))

        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=frame_reader, daemon=True).start()
    threading.Thread(target=start_caption_thread, daemon=True).start()
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        stop_event.set()
