# ESP32-CAM CaptionStream

A lightweight system that streams video from an ESP32-CAM and overlays real-time image captions using an LLM (e.g., OpenAI or Gemini). The captioned video is served via a local Flask web interface.

---

## Features

- Takes video stream from ip address (experimented with ESP32-CAM)
- Object recognition use YOLO model
- Use the YOLO result to hint LLM image caption
- View live captioned stream in a browser

---

## Prerequisite

Flash your ESP32 with Arduino's pre-set ESP32 CameraSever code, you can follow this [tutorial](https://randomnerdtutorials.com/esp32-cam-video-streaming-web-server-camera-home-assistant/).

The code to be flashed into camera can be found here: [https://github.com/espressif/arduino-esp32/tree/master/libraries/ESP32/examples/Camera/CameraWebServer]

The camera should connect to your wifi and streaming to your local ip.

---

## Setup

1. **Install dependencies**:
   ```bash
   pip install flask opencv-python aiohttp requests numpy Pillow
    ```
  2. **Configure your settings** in `captioned_web_stream.py`:

   ```python
   ESP32_STREAM_URL = "http://<your_esp32_ip>:81/stream"
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

---
