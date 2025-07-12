# ESP32-CAM CaptionStream

A lightweight system that streams video from an ESP32-CAM and overlays real-time image captions using an LLM (e.g., OpenAI or Gemini). The captioned video is served via a local Flask web interface.

---

## Features

- 📷 Streams MJPEG video from ESP32-CAM
- 🧠 Generates captions using OpenAI or Gemini APIs
- 🌐 View live captioned stream in a browser (`http://localhost:5050`)
- 🔁 Reconnects automatically on stream failure
- 🧵 Runs captioning in the background using `asyncio`

---

## Setup

1. **Install dependencies**:
   ```bash
   pip install flask opencv-python aiohttp requests numpy Pillow
    ```
  2. **Configure your settings** in `captioned_web_stream.py`:

   ```python
   ESP32_STREAM_URL = "http://<your_esp32_ip>:81/stream"
   CAPTION_API_URL = "https://localhost/tinytalk/api/image/caption" # endpoint for image caption
   MODEL = "openai"  # or "gemini"
   ```

3. **Run the app**:

   ```bash
   python captioned_web_stream.py
   ```

A window should pop up with your video steams and caption.

## Alternatively

Use flask app to start a web sever:
```python
python app.py
```

Then go to `http://127.0.0.1:5000` to check the video and caption.

---

## Notes

* Wi-Fi credentials for ESP32-CAM should be moved to `secrets.h` and excluded via `.gitignore`.
* Requires an existing caption API endpoint accepting multipart image input.

---
