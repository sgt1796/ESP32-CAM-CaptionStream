"""
Flask web-demo for ESP32-CAM video streaming + periodic image captioning

$ pip install flask aiohttp requests opencv-python pillow numpy
$ python app.py
"""
import asyncio
import threading
import time
import base64
from textwrap import wrap

import aiohttp
import cv2
import numpy as np
import requests
from flask import Flask, Response, render_template_string

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ESP32_STREAM_URL = "http://192.168.0.199:81/stream"
CAPTION_API_URL = None #"https://localhost/tinytalk/api/image/caption"
CAPTION_INTERVAL = 2        # seconds
MODEL            = "openai" # or "gemini"
VERIFY_SSL       = False    # self-signed TLS on localhost

llm = None
if not CAPTION_API_URL:
    # initialise LLM client
    from llm import LLM
    llm = LLM(model=MODEL)  # "openai" or "gemini"

# ─── GLOBAL STATE ──────────────────────────────────────────────────────────────
latest_jpeg   : bytes = None   # raw JPEG from camera
latest_caption: str   = ""     # last caption text
stop_event            = threading.Event()

# ─── BACKGROUND: MJPEG FRAME GRABBER ───────────────────────────────────────────
def mjpeg_reader() -> None:
    """Continuously read frames from ESP32-CAM and store latest_jpeg."""
    global latest_jpeg
    while not stop_event.is_set():
        try:
            print("[INFO] Connecting to ESP32 stream …")
            stream = requests.get(ESP32_STREAM_URL, stream=True, timeout=10)
            buf = b""
            for chunk in stream.iter_content(chunk_size=1024):
                buf += chunk
                start, end = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
                if start != -1 and end != -1:             # found a full JPEG
                    latest_jpeg = buf[start:end+2]
                    buf         = buf[end+2:]
        except Exception as e:
            print(f"[WARN] Stream error: {e}. Reconnecting in 2 s …")
            time.sleep(2)

# ─── BACKGROUND: PERIODIC CAPTION RETRIEVER ────────────────────────────────────
async def caption_worker() -> None:
    """Every CAPTION_INTERVAL s send latest frame to caption API."""
    global latest_caption, latest_jpeg
    if CAPTION_API_URL:
        connector = aiohttp.TCPConnector(ssl=VERIFY_SSL is True)
        async with aiohttp.ClientSession(connector=connector) as session:
            while not stop_event.is_set():
                await asyncio.sleep(CAPTION_INTERVAL)
                if not latest_jpeg:
                    continue
                try:
                    data = aiohttp.FormData()
                    data.add_field("base64jpg", "")                          # kept for API compatibility
                    data.add_field("model", MODEL)
                    # send binary JPEG
                    data.add_field("image_file", latest_jpeg,
                                filename="frame.jpg",
                                content_type="image/jpeg")
                    async with session.post(CAPTION_API_URL, data=data) as r:
                        if r.status == 200:
                            latest_caption = (await r.text()).strip()
                            print(f"[Caption] {latest_caption}")
                        else:
                            err = await r.text()
                            print(f"[ERROR {r.status}] {err}")
                except Exception as e:
                    print(f"[Caption exception] {e}")
    else:
        while not stop_event.is_set():
            await asyncio.sleep(CAPTION_INTERVAL)
            if not latest_jpeg:
                continue
            try:
                # Convert image to base64 for LLM input
                b64jpg = base64.b64encode(latest_jpeg).decode("utf-8")
                if MODEL == "openai":
                    result = llm.image_caption(imgbase64=b64jpg)
                else:
                    result = llm.image_caption(imgbase64=b64jpg, task = "caption")
                latest_caption = result.strip()
                print(f"[Caption LLM] {latest_caption}")
            except Exception as e:
                print(f"[Caption LLM Exception] {e}")

def start_caption_loop():
    """Run the async caption worker inside its own OS thread."""
    asyncio.run(caption_worker())

# ─── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!doctype html>
<title>ESP32-CAM Demo</title>
<style>
 body{background:#111;color:#eee;font-family:Arial,Helvetica,sans-serif;text-align:center}
 img{max-width:100%;height:auto;border:4px solid #444;border-radius:8px}
 .caption{margin-top:8px;font-family:monospace;font-size:1rem;white-space:pre-wrap}
</style>
<h1>ESP32-CAM Live Stream + Caption</h1>
<img src="{{ url_for('video_feed') }}">
<div class="caption" id="cap"></div>
<script>
 async function poll(){
   const r = await fetch("/caption", {cache:"no-store"});
   if(r.ok) document.getElementById("cap").textContent = await r.text();
 }
 setInterval(poll, {{ interval_ms }});
 poll();
</script>
"""

@app.route("/")
def index():
    return render_template_string(HTML, interval_ms=CAPTION_INTERVAL*1000)

@app.route("/caption")
def caption():
    # small helper endpoint for JS polling
    return latest_caption or "", 200, {"Cache-Control": "no-store"}

def mjpeg_generator():
    """Yield JPEG frames with caption text burned in (multipart/x-mixed-replace)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.6, 2
    while True:
        if latest_jpeg:
            # decode
            arr   = np.frombuffer(latest_jpeg, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            # overlay caption
            #if latest_caption:
            #    max_w = 40               # char per line before wrapping
            #    for i, line in enumerate(wrap(latest_caption, max_w)[:3]):
            #        y = 30 + i*30
            #        cv2.putText(frame, line, (10, y), font, scale,
            #                    (255,255,255), thick, cv2.LINE_AA)
            
            # encode
            ok, jpg = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
        else:
            time.sleep(0.01)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # start background threads
    threading.Thread(target=mjpeg_reader , daemon=True).start()
    threading.Thread(target=start_caption_loop, daemon=True).start()

    try:
        # threaded=True lets Flask handle multiple requests in dev mode
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        stop_event.set()
