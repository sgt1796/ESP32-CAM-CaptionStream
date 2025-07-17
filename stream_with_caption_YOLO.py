"""
ESP32-CAM real-time demo
• live MJPEG stream  • YOLOv10 object tracking  • periodic LLM captioning

$ pip install ultralytics opencv-python requests numpy
"""

import asyncio, base64, queue, threading, time

import cv2
import numpy as np
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError
from ultralytics import YOLO
from collections import Counter
import torch                              # for auto device detect

from llm import LLM                       # your own wrapper

# ─── CONFIG ──────────────────────────────────────────────────────────────
ESP32_STREAM_URL   = "http://192.168.0.199:81/stream"
YOLO_MODEL_PATH    = "yolo11n.pt"   # tiny INT8 → fastest
YOLO_IMGSZ         = 416
YOLO_CONF_THRESH   = 0.35
CAPTION_INTERVAL   = 2.0                  # seconds between captions
CAPTION_MODEL      = "openai"              # or "gemini"
CAPTION_AREA_H     = 110                  # px strip under the video
QUEUE_SIZE         = 1                    # keep only the freshest frame
USE_THUMBNAIL      = False                # set True current cause caption to fail

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────
frame_q: queue.Queue[bytes] = queue.Queue(maxsize=QUEUE_SIZE)
latest_jpeg: bytes | None   = None
jpeg_lock                   = threading.Lock()
caption: str                = ""
caption_lock            = threading.Lock()   # protects `caption`
scene_sig_latest        = None               # updated every frame
scene_sig_captioned     = None               # last signature we captioned
yolo_lock               = threading.Lock()   # protects sig + hints
yolo_objs_latest: list[tuple[str, float]] = []   # [(label, conf), …]


# ─── HEAVY OBJECTS (load once) ────────────────────────────────────────────
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")
yolo = YOLO(YOLO_MODEL_PATH)
yolo.overrides["conf"] = YOLO_CONF_THRESH        # set threshold once
llm  = LLM(model=CAPTION_MODEL)

# ─── STREAM READER THREAD ────────────────────────────────────────────────
def stream_reader() -> None:
    """Continuously push the newest JPEG into frame_q and latest_jpeg."""
    global latest_jpeg

    while True:
        try:
            print("[INFO] Connecting to ESP32-CAM stream …")
            resp = requests.get(ESP32_STREAM_URL, stream=True, timeout=8)
            buf = b""
            for chunk in resp.iter_content(chunk_size=1024):
                buf += chunk
                start = buf.find(b"\xff\xd8"); end = buf.find(b"\xff\xd9")
                if start != -1 and end != -1:
                    jpeg, buf = buf[start:end + 2], buf[end + 2:]

                    # —— store for display (queue) ——
                    if frame_q.full():
                        try: frame_q.get_nowait()
                        except queue.Empty: pass
                    frame_q.put_nowait(jpeg)

                    # —— store for caption (single slot) ——
                    with jpeg_lock:
                        latest_jpeg = jpeg
        except (ChunkedEncodingError, ConnectionError,
                requests.exceptions.ReadTimeout) as e:
            print(f"[WARN] stream error: {e} – reconnecting …")
            time.sleep(1)

# ─── CAPTION TASK ─────────────────────────────────────────────────────────
async def caption_loop() -> None:
    global caption, scene_sig_captioned 
    scene_sig_captioned = None  # reset on start
    while True:
        await asyncio.sleep(CAPTION_INTERVAL)

        with jpeg_lock:
            jpeg = latest_jpeg

        # no frame yet or likely corrupted → skip
        if not jpeg or len(jpeg) < 1000:
            continue

        try:
            THUMB_MIN_SIDE   = 256          # don’t go below this
            THUMB_JPEG_QUAL  = 80           # 70-85 is fine

            if USE_THUMBNAIL:
                arr   = np.frombuffer(jpeg, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue                       # bad JPEG, skip caption

                h, w = frame.shape[:2]
                # Only shrink if the frame is larger than the safe minimum
                if min(h, w) > THUMB_MIN_SIDE:
                    scale   = THUMB_MIN_SIDE / min(h, w)
                    new_w   = int(w * scale)
                    new_h   = int(h * scale)
                    frame   = cv2.resize(frame, (new_w, new_h),
                                        interpolation=cv2.INTER_AREA)

                ok, buf = cv2.imencode(".jpg", frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), THUMB_JPEG_QUAL])
                if not ok:                       # fallback on rare encode failure
                    buf = arr
                b64 = base64.b64encode(buf).decode()
            else:
                b64 = base64.b64encode(jpeg).decode()

            # ----- build detector hint list (thread-safe read) -----
            with yolo_lock:
                objs   = yolo_objs_latest.copy()
                sig    = scene_sig_latest

            if sig == scene_sig_captioned:
                continue        # scene unchanged
            
            print(f"[Caption] new scene signature: {sig} | objects: {objs} | Calling LLM …")
            # ----- call the caption with object dection func -----
            obj_counts = [(o["label"], o["count"]) for o in objs]
            with caption_lock:
                caption = llm.caption_with_objects(
                    imgbase64=b64,
                    objects=obj_counts,
                ).strip()
            scene_sig_captioned = sig  # remember captioned signature
            #caption = llm.image_caption(imgbase64=b64).strip()
            print(f"[Caption] {caption}")
        except Exception as e:
            print(f"[Caption] error: {e}")

# ─── INFERENCE + DISPLAY LOOP ────────────────────────────────────────────
def infer_and_show() -> None:
    font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1
    txt_col, bg_col = (255, 255, 255), (0, 0, 0)

    while True:
        jpeg = frame_q.get()                      # blocks for newest frame
        arr  = np.frombuffer(jpeg, np.uint8)
        if not arr.size:
            continue                              # guard empty buffer
        try:
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except cv2.error:
            continue
        if frame is None:
            continue

        # — YOLO tracking —
        try:
            res = yolo.track(frame, imgsz=YOLO_IMGSZ,
                            persist=True, verbose=False, device=DEVICE)[0]
            frame = res.plot()

            # —— build signature & hint ————————————————
            #  a) one highest-confidence box per class
            with yolo_lock:
                best_per_class = {}
                for cls, conf in zip(res.boxes.cls.cpu().tolist(),
                                    res.boxes.conf.cpu().tolist()):
                    cls = int(cls)
                    if cls not in best_per_class or conf > best_per_class[cls]:
                        best_per_class[cls] = conf

                global scene_sig_latest, yolo_objs_latest
                cls_list  = [int(c) for c in res.boxes.cls.cpu().tolist()]
                cls_count = Counter(cls_list)              # {class: n_boxes}
                scene_sig_latest = tuple(sorted(cls_count.items()))
                # send label, **count**, best-confidence
                yolo_objs_latest = [
                    {
                        "label":  yolo.model.names[c],
                        "count":  cls_count[c],
                        "conf":   best_per_class[c],
                    }
                    for c in sorted(best_per_class)
                ]
                #print(f"[YOLO] {yolo_objs_latest} | sig: {scene_sig_latest}") # debug

        except Exception as e:
            print(f"[YOLO] {e}")

        # — caption strip —
        h, w = frame.shape[:2]
        cap_bg = np.full((CAPTION_AREA_H, w, 3), bg_col, np.uint8)
        if caption:
            lines, line = [], ""
            for word in caption.split():
                if len(line) + len(word) < 60:
                    line += word + " "
                else:
                    lines.append(line); line = word + " "
            lines.append(line)
            for i, ln in enumerate(lines[:4]):          # max 4 lines
                cv2.putText(cap_bg, ln.strip(), (10, 22 + i * 20),
                            font, fscale, txt_col, thick, cv2.LINE_AA)

        cv2.imshow("ESP32-CAM | YOLOv10 | LLM Caption", np.vstack((frame, cap_bg)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

# ─── MAIN (async) ─────────────────────────────────────────────────────────
async def main() -> None:
    threading.Thread(target=stream_reader, daemon=True).start()
    cap_task = asyncio.create_task(caption_loop())
    await asyncio.to_thread(infer_and_show)
    cap_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
