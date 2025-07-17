"""
Flask web-UI for ESP32-CAM + YOLO + LLM caption demo
Run with:  python app.py
Open  http://localhost:5000  in a browser.
"""

import threading, asyncio, cv2, numpy as np
from flask import Flask, Response, render_template, jsonify
from collections import Counter

# ── re-use your existing globals ─────────────────────────────────────────
import stream_with_caption_YOLO as cam

stream_reader   = cam.stream_reader
caption_loop    = cam.caption_loop
frame_q         = cam.frame_q
yolo            = cam.yolo
YOLO_IMGSZ      = cam.YOLO_IMGSZ
DEVICE          = cam.DEVICE
caption_lock    = cam.caption_lock

# Flask app
app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Frame generator  →  multipart/x-mixed-replace stream
# ─────────────────────────────────────────────────────────────────────────
def mjpeg_generator():
    font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1
    txt_col, bg_col = (255, 255, 255), (0, 0, 0)
    CAPTION_AREA_H = 110                    # same as original script

    while True:
        jpeg = frame_q.get()          # or however you fetch the JPEG
        if not jpeg:
            continue                  # empty bytes – skip

        arr = np.frombuffer(jpeg, np.uint8)
        if arr.size == 0:
            continue                  # still empty – skip

        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue                  # corrupt – skip
        if frame is None:
            continue

        # YOLO tracking (same call you used before)
        res   = yolo.track(frame, imgsz=YOLO_IMGSZ,
                           persist=True, verbose=False,
                           device=DEVICE)[0]
        frame = res.plot()
        # ── STEP 1: compute a stable “scene signature” + object list ──
        best_per_class = {}
        for cls, conf in zip(res.boxes.cls.cpu().tolist(),
                             res.boxes.conf.cpu().tolist()):
            cls = int(cls)
            if cls not in best_per_class or conf > best_per_class[cls]:
                best_per_class[cls] = conf

        cls_list  = [int(c) for c in res.boxes.cls.cpu().tolist()]
        cls_count = Counter(cls_list)                         # {class: n}

        # ── STEP 2: hand them to caption_loop() in a threadsafe way ──
        with cam.yolo_lock:
            cam.scene_sig_latest = tuple(sorted(cls_count.items()))
            cam.yolo_objs_latest = [
                {
                    "label": yolo.model.names[c],
                    "count": cls_count[c],
                    "conf":  best_per_class[c],
                }
                for c in sorted(best_per_class)
            ]

        # Draw caption (if any)
        with caption_lock:
            txt = cam.caption
        if txt:
            h, w = frame.shape[:2]
            cap_bg = np.full((CAPTION_AREA_H, w, 3), bg_col, np.uint8)
            # crude line-wrapping (≤60 chars/line)
            y = 22; line = ""
            for word in txt.split():
                if len(line) + len(word) < 60:
                    line += word + " "
                else:
                    cv2.putText(cap_bg, line.strip(), (10, y),
                                font, fscale, txt_col, thick, cv2.LINE_AA)
                    y += 20; line = word + " "
            cv2.putText(cap_bg, line.strip(), (10, y),
                        font, fscale, txt_col, thick, cv2.LINE_AA)
            frame = np.vstack((frame, cap_bg))

        # Encode back to JPEG
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")


# ── Flask routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Simple HTML page with the video element."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream; <img src> points here."""
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/caption")
def caption_api():
    with caption_lock:
        txt = cam.caption
    return jsonify({"caption": txt})


# ── background threads ──────────────────────────────────────────────────
def run_caption_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(caption_loop())


if __name__ == "__main__":
    # start ESP32 reader + caption task in background
    threading.Thread(target=stream_reader, daemon=True).start()
    threading.Thread(target=run_caption_loop, daemon=True).start()

    # launch Flask (threaded=True so generator keeps up)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
