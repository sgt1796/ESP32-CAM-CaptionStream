import asyncio
import cv2
import time
import numpy as np
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError

## For image captioning
from llm import LLM
import base64


ESP32_STREAM_URL = "http://192.168.0.199:81/stream"
CAPTION_INTERVAL = 2  # seconds
MODEL = "openai" # or "gemini"
llm = LLM(model=MODEL)


caption = ""
latest_jpeg = None
verify_ssl = False

def read_mjpeg_stream():
    """Generator to yield JPEG bytes from MJPEG stream with auto-reconnect."""
    while True:
        try:
            print("[INFO] Connecting to ESP32 stream...")
            stream = requests.get(ESP32_STREAM_URL, stream=True, timeout=10)
            byte_buffer = b""
            for chunk in stream.iter_content(chunk_size=1024):
                byte_buffer += chunk
                start = byte_buffer.find(b'\xff\xd8')
                end = byte_buffer.find(b'\xff\xd9')
                if start != -1 and end != -1:
                    jpg = byte_buffer[start:end+2]
                    byte_buffer = byte_buffer[end+2:]
                    yield jpg
        except (ChunkedEncodingError, ConnectionError, requests.exceptions.ReadTimeout) as e:
            print(f"[WARN] Stream disconnected: {e}. Reconnecting in 2s...")
            time.sleep(2)
        except Exception as e:
            print(f"[ERROR] Unexpected stream failure: {e}")
            time.sleep(2)

async def caption_loop():
    global caption, latest_jpeg
    while True:
        await asyncio.sleep(CAPTION_INTERVAL)
        if latest_jpeg:
            try:
                # Convert JPEG bytes to base64 string
                b64jpg = base64.b64encode(latest_jpeg).decode("utf-8")

                # Direct call: for OpenAI, you can use imgbase64
                if MODEL == "openai":
                    # Use LLM image_caption function directly
                    result = llm.image_caption(imgbase64=b64jpg)
                else:
                    # For Gemini, also support imgbase64 or PIL image
                    result = llm.image_caption(imgbase64=b64jpg)

                caption = result.strip()
                print(f"[Caption] {caption}\n")
            except Exception as e:
                print(f"[Caption Exception] {e}\n")


def run_display_loop():
    global latest_jpeg, caption
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    caption_area_height = 120
    text_color = (255, 255, 255)  # white
    bg_color = (0, 0, 0)          # black

    for jpeg_bytes in read_mjpeg_stream():
        latest_jpeg = jpeg_bytes
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        if arr.size == 0:
            continue
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        caption_area = np.full((caption_area_height, w, 3), bg_color, dtype=np.uint8)

        if caption:
            # Split caption into multiple lines if too long
            wrapped_lines = []
            max_line_length = 60
            words = caption.split()
            line = ""
            for word in words:
                if len(line + word) < max_line_length:
                    line += word + " "
                else:
                    wrapped_lines.append(line.strip())
                    line = word + " "
            wrapped_lines.append(line.strip())

            for i, line in enumerate(wrapped_lines[:5]):  # limit to 2 lines max
                y = 20 + i * 20
                cv2.putText(caption_area, line, (10, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Stack vertically: video + caption area
        combined = np.vstack((frame, caption_area))

        cv2.imshow("ESP32 Stream + Caption", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


async def main():
    caption_task = asyncio.create_task(caption_loop())
    await asyncio.to_thread(run_display_loop)
    caption_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
