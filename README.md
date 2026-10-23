# Indoor Drone Vision

Real-time, responsive object detection and ground-station vision telemetry for a manually RC-controlled indoor drone.

> **Safety:** Observational vision and monitoring only.
> This software does **not** control the flight controller, motors, RC inputs, MAVLink, or serial telemetry ports.
> The drone remains entirely under manual RC pilot control.

---

## 1. System Specifications & Architecture

| Component | Detail |
|---|---|
| **SBC** | Raspberry Pi 4 Model B, nominal 4 GB RAM |
| **OS** | Debian 13 (Trixie), ARM64 (Linux 6.18.50 PREEMPT) |
| **Python** | 3.13.5 (`.venv`) |
| **Camera** | Mi USB Webcam HD (`/dev/video0`, UVC/V4L2) |
| **Capture Resolution** | 640×480 MJPG @ 30 FPS |
| **Detection Engine** | ONNX Runtime 1.31.0 (`CPUExecutionProvider`, 2 intra-op threads) |
| **Default Model** | MobileNet-SSD v1 INT8 (`ssd_mobilenet_v1_int8.onnx`, ~254 ms / 3.9 FPS) |
| **Fallback Model** | MobileNet-SSD v1 FP32 (`ssd_mobilenet_v1.onnx`, ~423 ms / 2.4 FPS) |
| **Ground Station Web** | Python `ThreadingHTTPServer` (zero extra dependencies, default `127.0.0.1:8080`) |

### Decoupled Real-Time Vision Pipeline
```
[USB Camera /dev/video0]
       │ (V4L2 MJPG ~30 FPS)
       ▼
 [CaptureWorker Thread]
       │
       ▼ (Bounded Queue, capacity=1, drop oldest)
 [LatestFrameBuffer]
       │ (Fresh frame, ~150ms frame age)
       ▼
[InferenceWorker Thread] ───► [ObjectDetector (INT8, 2 threads)]
       │ (Inference: ~250-295ms, 3.6-4.0 FPS)
       ▼
[LatestResultBuffer]
       │
       ├──► [GroundStationServer (MJPEG /stream + /api/telemetry)]
       └──► [Local Display Window / HUD (if GUI available)]
```

---

## 2. Quick Start Commands

Activate the virtual environment first:
```bash
source .venv/bin/activate
```

### A. Run Decoupled Detection with Ground-Station Dashboard
```bash
# Starts camera capture, INT8 inference, and local dashboard at http://127.0.0.1:8080
python -m src.main --detect --headless --stream
```

### B. Access Ground-Station from Another Device on Local Wi-Fi
To allow browser monitoring from a laptop/tablet on the same trusted private Wi-Fi:
```bash
python -m src.main --detect --headless --stream --host 0.0.0.0 --port 8080
```
Then navigate to `http://<pi-ip-address>:8080` in any web browser.

### C. Headless Test Verification Modes (Phase 1 & Phase 2 Compatible)
```bash
# Phase 1: Camera capture test (10 frames, camera-only)
python -m src.main --headless --test

# Phase 2: Synchronous detection test (10 frames with model inference & snapshots)
python -m src.main --headless --test --detect
```

---

## 3. Ground Station Dashboard & Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Responsive dark-mode Ground Station web dashboard with live video, session telemetry, object tables with apparent-size (% area), and stale-stream alert |
| `/stream` | `GET` | Multipart MJPEG continuous video stream (`multipart/x-mixed-replace`) |
| `/snapshot` | `GET` | Single JPEG snapshot of the latest annotated frame |
| `/api/telemetry` | `GET` | JSON endpoint exposing `session_id`, `elapsed_s`, `capture_fps`, `inference_fps`, `inference_latency_ms`, `frame_age_ms`, `dropped_frames`, `temperature_c`, `throttled_hex`, `memory_available_mb`, `is_camera_connected`, `is_stale`, and detected objects list |

---

## 4. Session Logging & Observational Telemetry

The system includes a dedicated, non-blocking `SessionLogger` (`src/session_logger.py`) designed to minimize SD card write wear on the Raspberry Pi:
- **Unique Session IDs**: Generated on startup (format: `session_YYYYMMDD_HHMMSS_<rand4>`).
- **Detection Event Logging (CSV)**: Recorded to `output/sessions/<session_id>_detections.csv`. Captures ISO/epoch timestamps, frame ID, class ID/name, confidence score, bounding box pixel coordinates, and **apparent size (% image area)**.
- **Bounded Worker Queue**: Detection events pass through a bounded memory queue (default capacity: 1,000 items). If disk I/O stalls, events are dropped safely without blocking camera capture or inference threads.
- **Periodic Disk Flushes**: Buffers are flushed every 2.0 seconds (`flush_interval_s`) rather than per-frame.
- **Session Summary (JSON)**: Written to `output/sessions/<session_id>_summary.json` upon clean shutdown. Aggregates duration, model metadata, capture/inference totals, dropped frame count, average frame rates and latencies, ending SoC temperature, available RAM, throttling hex flags, and class detection frequency.
- **Privacy & Storage Protection**: No raw video footage is stored on disk by default. Annotated frame snapshots remain optional (`save_snapshots: false`).
- **Apparent-Size Indicator**: Bounding box pixel area relative to the frame is reported as `apparent_size_pct` ("image area %").
  > **Safety Notice:** Apparent size is strictly a 2D bounding-box pixel area ratio on the camera image plane. It is **never** an accurate distance, proximity, or collision risk metric, nor does it provide a collision avoidance guarantee. It never triggers drone maneuvers or flight decisions.

---

## 5. Security & Network Restrictions

- **Default Binding**: Binds strictly to `127.0.0.1` (localhost).
- **LAN Access (`0.0.0.0`)**: Only enable on a trusted, private, password-protected local Wi-Fi network.
- **Never Public Internet**: Never forward router ports (NAT) or expose this server to the public internet.
- **Strict Read-Only**: The web server provides strictly observational telemetry. It has **no** flight control, arming, disarming, motor, MAVLink, or parameter modification endpoints.

---

## 6. Running the Test Suite

The test suite covers label mapping, detector preprocessing/decoding/NMS, model integration, decoupled pipeline thread-safety, ground-station server endpoints, and session logging:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

---

## 7. Project Structure

```
indoor_drone_vision/
├── config.yaml          # Camera, detector, pipeline, server, and session_logger settings
├── requirements.txt     # Python dependencies
├── .gitignore
├── README.md            # Project overview and quick start
├── models/
│   ├── README.md        # Provenance, licenses, and benchmark comparisons
│   ├── ssd_mobilenet_v1_int8.onnx  # Quantized detector (9.1 MB, 254 ms)
│   └── ssd_mobilenet_v1.onnx       # FP32 detector fallback (29.4 MB, 423 ms)
├── src/
│   ├── __init__.py
│   ├── camera.py        # Camera capture module (V4L2 MJPG)
│   ├── detector.py      # ONNX detection engine (preprocessing, postprocessing, NMS, apparent-size)
│   ├── labels.py        # COCO 91-category mapping & indoor target classes
│   ├── pipeline.py      # Decoupled CaptureWorker, InferenceWorker & LatestFrameBuffer
│   ├── server.py        # GroundStationServer & MJPEG streaming handler with stale alert
│   ├── session_logger.py # Thread-safe persistent session CSV & JSON logging
│   └── main.py          # Unified CLI and orchestration entry point
├── tests/
│   ├── __init__.py
│   ├── test_detector.py # Unit & integration tests for detection engine & apparent size
│   ├── test_pipeline.py # Unit tests for decoupled pipeline & stale-frame dropping
│   ├── test_server.py   # Unit tests for HTTP dashboard, telemetry, & stream
│   ├── test_session_logger.py # Unit tests for session ID, CSV/JSON lifecycle, & queue bounds
│   └── test_person.jpg  # Standard test image asset
└── output/
    ├── sessions/        # Persistent session CSV logs & JSON summaries
    ├── snapshots/       # Annotated detection snapshot frames (optional)
    └── *.jpg            # Raw camera test frames
```

---

## 8. Project Phases

- [x] **Phase 1** — Camera capture foundation (V4L2 MJPG @ 30+ FPS)
- [x] **Phase 2** — Pretrained object detection integration (SSD MobileNet v1 via ONNX Runtime)
- [x] **Phase 3** — Responsive multithreaded pipeline & Ground-Station web streaming
- [x] **Phase 4** — Session logging, Ground-Station reliability & apparent-size indicators
