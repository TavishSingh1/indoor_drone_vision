"""
server.py — Lightweight ground-station telemetry & MJPEG web streaming server.

Provides a read-only browser dashboard for observational monitoring over Wi-Fi.
Uses Python's standard library ThreadingHTTPServer — zero extra dependencies.

Endpoints:
- GET /               : Interactive Ground-Station monitoring dashboard (HTML/JS)
- GET /stream         : Multipart MJPEG live video stream
- GET /snapshot       : Latest annotated frame as a single JPEG
- GET /api/telemetry  : Current vision metrics, detections, and Pi SoC telemetry (JSON)

Safety:
- Read-only vision telemetry and camera stream.
- Zero flight-control, arming, motor, MAVLink or RC endpoints.
- Binds to 127.0.0.1 by default. Explicit configuration required for LAN interface (0.0.0.0).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

from src.pipeline import VisionPipeline

logger = logging.getLogger(__name__)

# Embedded clean, dark-mode, responsive HTML/CSS/JS dashboard
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Indoor Drone Vision — Ground Station</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #10b981;
      --warn: #f59e0b;
      --danger: #ef4444;
      --hud-blue: #38bdf8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 16px;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 14px;
    }
    h1 { font-size: 1.2rem; font-weight: 700; display: flex; align-items: center; gap: 8px; }
    .badge {
      font-size: 0.75rem;
      padding: 3px 8px;
      border-radius: 9999px;
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent);
      border: 1px solid var(--accent);
      font-weight: 600;
    }
    .session-pill {
      font-size: 0.75rem;
      font-family: monospace;
      padding: 3px 10px;
      border-radius: 6px;
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--text-muted);
    }
    #stale-alert {
      display: none;
      background: rgba(239, 68, 68, 0.2);
      border: 1px solid var(--danger);
      color: #fca5a5;
      padding: 10px 14px;
      border-radius: 6px;
      margin-bottom: 14px;
      font-weight: 600;
      font-size: 0.9rem;
      align-items: center;
      gap: 8px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }
    @media (min-width: 960px) {
      .grid { grid-template-columns: 1.8fr 1.2fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    .video-container {
      position: relative;
      background: #000;
      border-radius: 6px;
      overflow: hidden;
      aspect-ratio: 4 / 3;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .video-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }
    .stat-box {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 10px;
    }
    .stat-label {
      font-size: 0.7rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .stat-val {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--hud-blue);
      margin-top: 2px;
    }
    .objects-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
      margin-top: 8px;
    }
    .objects-table th {
      text-align: left;
      color: var(--text-muted);
      padding: 6px;
      border-bottom: 1px solid var(--border);
      font-weight: 600;
    }
    .objects-table td {
      padding: 6px;
      border-bottom: 1px solid rgba(51, 65, 85, 0.4);
    }
    .tag {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--hud-blue);
      font-weight: 600;
    }
    .count-pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      background: rgba(248, 250, 252, 0.1);
      margin-right: 6px;
      margin-bottom: 6px;
      font-size: 0.75rem;
    }
    .disclaimer {
      font-size: 0.7rem;
      color: var(--text-muted);
      margin-top: 6px;
      line-height: 1.3;
    }
    footer {
      margin-top: 16px;
      font-size: 0.75rem;
      color: var(--text-muted);
      text-align: center;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>🚁 Indoor Drone Vision <span class="badge">GROUND STATION</span></h1>
      <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;">
        Observational Telemetry Dashboard • Pure Read-Only
      </div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
      <span class="session-pill" id="val-session-id">Session: initializing...</span>
      <span class="session-pill" id="val-elapsed">00:00:00</span>
      <span id="connection-status" style="font-size: 0.85rem; color: var(--accent); font-weight: 600;">● Live Stream</span>
    </div>
  </header>

  <div id="stale-alert">
    ⚠️ <strong>CAMERA STREAM STALE OR DISCONNECTED:</strong> No recent camera frames received. Check USB connection.
  </div>

  <div class="grid">
    <!-- Video Stream Panel -->
    <div class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h2 style="font-size: 0.95rem; color: var(--text-muted);">LIVE VIDEO STREAM</h2>
        <span style="font-size: 0.75rem; color: var(--text-muted);" id="val-model-tag">Model: --</span>
      </div>
      <div class="video-container">
        <img src="/stream" alt="Live Camera Stream" id="stream-img" onerror="handleStreamError()">
      </div>
      <div class="disclaimer">
        * Bounding boxes indicate 2D detection in camera frame. Apparent size reflects image area percentage only; it does not measure real-world distance or collision risk.
      </div>
    </div>

    <!-- Telemetry Panel -->
    <div class="card">
      <h2 style="font-size: 0.95rem; margin-bottom: 10px; color: var(--text-muted);">PIPELINE TELEMETRY</h2>
      
      <div class="telemetry-grid">
        <div class="stat-box">
          <div class="stat-label">Camera FPS</div>
          <div class="stat-val" id="val-cap-fps">-- FPS</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Inference FPS</div>
          <div class="stat-val" id="val-inf-fps">-- FPS</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Infer Latency</div>
          <div class="stat-val" id="val-latency">-- ms</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Frame Age</div>
          <div class="stat-val" id="val-frame-age">-- ms</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Dropped Frames</div>
          <div class="stat-val" id="val-dropped" style="color: var(--warn);">--</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">CPU Temp</div>
          <div class="stat-val" id="val-temp">-- °C</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Free RAM</div>
          <div class="stat-val" id="val-ram">-- MB</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Camera Status</div>
          <div class="stat-val" id="val-cam-status" style="font-size: 0.95rem;">--</div>
        </div>
      </div>

      <h2 style="font-size: 0.9rem; margin-top: 14px; margin-bottom: 6px; color: var(--text-muted);">ACTIVE OBJECTS & APPARENT SIZE</h2>
      <div id="class-counts-container" style="margin-bottom: 6px;"></div>

      <div style="max-height: 220px; overflow-y: auto;">
        <table class="objects-table">
          <thead>
            <tr>
              <th>Class</th>
              <th>Conf</th>
              <th>Apparent Size</th>
              <th>Box (x1, y1, x2, y2)</th>
            </tr>
          </thead>
          <tbody id="objects-body">
            <tr><td colspan="4" style="text-align:center; color:var(--text-muted);">No detections</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <footer>
    Indoor Drone Vision Ground Station • Observational telemetry only • Manual RC pilot control • Independent of flight controller
  </footer>

  <script>
    function formatSeconds(secs) {
      if (!secs || secs < 0) secs = 0;
      const h = Math.floor(secs / 3600);
      const m = Math.floor((secs % 3600) / 60);
      const s = Math.floor(secs % 60);
      return [h, m, s].map(v => v.toString().padStart(2, '0')).join(':');
    }

    function handleStreamError() {
      document.getElementById('connection-status').textContent = '○ Reconnecting stream...';
      document.getElementById('connection-status').style.color = 'var(--warn)';
      setTimeout(() => {
        document.getElementById('stream-img').src = '/stream?t=' + Date.now();
      }, 1000);
    }

    async function updateTelemetry() {
      try {
        const resp = await fetch('/api/telemetry');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();

        // Session & Header info
        if (data.session_id) {
          document.getElementById('val-session-id').textContent = data.session_id;
        }
        if (data.session_elapsed_s !== undefined) {
          document.getElementById('val-elapsed').textContent = formatSeconds(data.session_elapsed_s);
        }
        if (data.model_name) {
          document.getElementById('val-model-tag').textContent = 'Model: ' + data.model_name + ' (' + (data.model_status || 'Loaded') + ')';
        }

        // Telemetry numbers
        document.getElementById('val-cap-fps').textContent = (data.capture_fps || 0).toFixed(1) + ' FPS';
        document.getElementById('val-inf-fps').textContent = (data.inference_fps || 0).toFixed(1) + ' FPS';
        document.getElementById('val-latency').textContent = (data.inference_latency_ms || 0).toFixed(0) + ' ms';
        document.getElementById('val-frame-age').textContent = (data.frame_age_ms || 0).toFixed(0) + ' ms';
        document.getElementById('val-dropped').textContent = (data.dropped_frames || 0).toLocaleString();
        document.getElementById('val-temp').textContent = data.temperature_c ? (data.temperature_c.toFixed(1) + ' °C') : 'N/A';
        document.getElementById('val-ram').textContent = data.memory_available_mb ? (data.memory_available_mb.toFixed(0) + ' MB') : 'N/A';
        
        const camStatus = data.camera_status || 'Unknown';
        const camStatusEl = document.getElementById('val-cam-status');
        camStatusEl.textContent = camStatus;
        if (camStatus === 'Active') {
          camStatusEl.style.color = 'var(--accent)';
        } else if (camStatus === 'Stale') {
          camStatusEl.style.color = 'var(--warn)';
        } else {
          camStatusEl.style.color = 'var(--danger)';
        }

        // Stale warning banner
        const staleAlert = document.getElementById('stale-alert');
        if (data.is_stale || !data.is_camera_connected) {
          staleAlert.style.display = 'flex';
        } else {
          staleAlert.style.display = 'none';
        }

        // Per-class count pills
        const countsDiv = document.getElementById('class-counts-container');
        if (data.counts && Object.keys(data.counts).length > 0) {
          let pills = '';
          for (const [cls, cnt] of Object.entries(data.counts)) {
            pills += `<span class="count-pill"><strong>${cls}</strong>: ${cnt}</span>`;
          }
          countsDiv.innerHTML = pills;
        } else {
          countsDiv.innerHTML = '<span style="font-size:0.75rem; color:var(--text-muted);">No objects currently in view</span>';
        }

        // Objects table
        const tbody = document.getElementById('objects-body');
        if (!data.detections || data.detections.length === 0) {
          tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-muted);">No detections</td></tr>';
        } else {
          let rows = '';
          for (const det of data.detections) {
            const b = det.box;
            const sizePct = (det.apparent_size_pct !== undefined) ? det.apparent_size_pct.toFixed(1) + '% area' : '--';
            rows += `<tr>
              <td><span class="tag">${det.class_name}</span></td>
              <td>${(det.confidence * 100).toFixed(0)}%</td>
              <td style="color:var(--hud-blue); font-weight:600;">${sizePct}</td>
              <td style="color:var(--text-muted); font-family:monospace;">[${b[0]}, ${b[1]}, ${b[2]}, ${b[3]}]</td>
            </tr>`;
          }
          tbody.innerHTML = rows;
        }

        document.getElementById('connection-status').textContent = '● Live Stream';
        document.getElementById('connection-status').style.color = 'var(--accent)';
      } catch (e) {
        document.getElementById('connection-status').textContent = '○ Telemetry offline';
        document.getElementById('connection-status').style.color = 'var(--danger)';
      }
    }

    setInterval(updateTelemetry, 500);
    updateTelemetry();
  </script>
</body>
</html>
"""


class GroundStationRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for ground-station dashboard and stream."""

    server: "GroundStationServer"  # Type hint for server instance

    def log_message(self, format: str, *args) -> None:
        """Suppress default access log flooding from continuous MJPEG streaming."""
        pass

    def do_GET(self) -> None:
        """Route GET requests."""
        path = self.path.split("?")[0]

        if path == "/" or path == "/index.html":
            self._serve_dashboard()
        elif path == "/stream":
            self._serve_mjpeg_stream()
        elif path == "/snapshot":
            self._serve_snapshot()
        elif path == "/api/telemetry":
            self._serve_telemetry_json()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def _serve_dashboard(self) -> None:
        """Serve the single-page HTML ground station dashboard."""
        body = DASHBOARD_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _serve_telemetry_json(self) -> None:
        """Serve real-time vision telemetry, session info, and object detections as JSON."""
        pipeline = self.server.pipeline
        telemetry = pipeline.get_telemetry()
        result = pipeline.get_latest_result()

        detections_data = []
        counts_data = {}
        if result is not None:
            counts_data = result.counts
            for det in result.detections:
                detections_data.append(
                    {
                        "class_id": det.class_id,
                        "class_name": det.class_name,
                        "confidence": round(det.confidence, 3),
                        "apparent_size_pct": getattr(det, "apparent_size_pct", 0.0),
                        "box": list(det.box),
                    }
                )

        session_id = (
            pipeline.session_logger.session_id
            if pipeline.session_logger is not None
            else "none"
        )
        session_start = (
            pipeline.session_logger.start_epoch
            if pipeline.session_logger is not None
            else self.server.start_time
        )
        session_elapsed = round(time.time() - session_start, 1)

        camera_status = (
            "Disconnected"
            if not telemetry.is_camera_connected
            else ("Stale" if telemetry.is_stale else "Active")
        )
        model_name = "None"
        if pipeline.detector is not None:
            det_cfg = getattr(pipeline.detector, "config", None)
            if det_cfg and hasattr(det_cfg, "model_path"):
                model_name = det_cfg.model_path.split("/")[-1]
            else:
                model_name = getattr(pipeline.detector, "model_name", "Detector")
        model_status = (
            "Loaded"
            if (pipeline.detector is not None and pipeline.detector.is_loaded)
            else "Standby"
        )

        payload = {
            "timestamp": time.time(),
            "session_id": session_id,
            "session_elapsed_s": session_elapsed,
            "camera_status": camera_status,
            "model_name": model_name,
            "model_status": model_status,
            "is_stale": telemetry.is_stale,
            "is_camera_connected": telemetry.is_camera_connected,
            "capture_fps": round(telemetry.capture_fps, 1),
            "inference_fps": round(telemetry.inference_fps, 1),
            "inference_latency_ms": round(telemetry.inference_latency_ms, 1),
            "frame_age_ms": round(telemetry.frame_age_ms, 1),
            "end_to_end_latency_ms": round(telemetry.end_to_end_latency_ms, 1),
            "total_captured": telemetry.total_captured,
            "total_inferred": telemetry.total_inferred,
            "dropped_frames": telemetry.dropped_frames,
            "temperature_c": telemetry.temperature_c,
            "memory_available_mb": telemetry.memory_available_mb,
            "throttled_hex": telemetry.throttled_hex,
            "counts": counts_data,
            "detections": detections_data,
        }

        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_snapshot(self) -> None:
        """Serve latest annotated frame as a single JPEG image."""
        frame = self.server.pipeline.get_latest_frame(annotated=True)
        if frame is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No frame available yet")
            return

        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.server.jpeg_quality],
        )
        if not ok:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "JPEG encode failed")
            return

        jpeg_bytes = buf.tobytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg_bytes)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(jpeg_bytes)

    def _serve_mjpeg_stream(self) -> None:
        """Stream latest frames as multipart/x-mixed-replace."""
        self.send_response(HTTPStatus.OK)
        boundary = "frame"
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={boundary}",
        )
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        logger.info("Ground-station client connected to /stream: %s", self.client_address)

        try:
            last_sent_time = 0.0
            min_interval = 1.0 / self.server.stream_max_fps

            while not self.server.is_stopping:
                now = time.monotonic()
                if now - last_sent_time < min_interval:
                    time.sleep(0.005)
                    continue

                frame = self.server.pipeline.get_latest_frame(annotated=True)
                if frame is None:
                    time.sleep(0.02)
                    continue

                ok, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.server.jpeg_quality],
                )
                if not ok:
                    continue

                jpeg_bytes = buf.tobytes()
                part_header = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
                ).encode("ascii")

                self.wfile.write(part_header)
                self.wfile.write(jpeg_bytes)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                last_sent_time = now

        except (BrokenPipeError, ConnectionResetError):
            logger.info("Ground-station client disconnected: %s", self.client_address)
        except Exception as e:
            logger.debug("Stream worker exception: %s", e)


class GroundStationServer(ThreadingHTTPServer):
    """Threaded HTTP server hosting ground station telemetry and stream."""

    def __init__(
        self,
        host: str,
        port: int,
        pipeline: VisionPipeline,
        jpeg_quality: int = 80,
        stream_max_fps: int = 25,
    ) -> None:
        super().__init__((host, port), GroundStationRequestHandler)
        self.pipeline = pipeline
        self.jpeg_quality = jpeg_quality
        self.stream_max_fps = stream_max_fps
        self.start_time = time.time()
        self._is_stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_stopping(self) -> bool:
        return self._is_stopping.is_set()

    def start_background(self) -> None:
        """Start the server listening in a daemon thread."""
        self._is_stopping.clear()
        self._thread = threading.Thread(
            target=self.serve_forever,
            name="GroundStationServer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Ground Station Server active at http://%s:%d (default bind: localhost)",
            self.server_address[0],
            self.server_port,
        )

    def stop(self) -> None:
        """Stop server gracefully."""
        self._is_stopping.set()
        logger.info("Shutting down Ground Station Server...")
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Ground Station Server stopped")
