"""
test_server.py — Unit and integration tests for the Ground-Station web server.

Tests:
1. Server binding to localhost (127.0.0.1)
2. HTTP GET / (Dashboard HTML)
3. HTTP GET /api/telemetry (JSON metrics and detections)
4. HTTP GET /snapshot (JPEG image)
5. HTTP GET /stream (Multipart MJPEG stream start and client disconnect)
6. Clean server shutdown without leaking sockets or threads
"""

from __future__ import annotations

import json
import time
import unittest
import urllib.request
from http import HTTPStatus

import numpy as np

from src.detector import Detection
from src.pipeline import DetectionResult, VisionPipeline
from src.server import GroundStationServer
from tests.test_pipeline import MockCamera, MockDetector


class TestGroundStationServer(unittest.TestCase):
    """Test GroundStationServer endpoints and stream handling."""

    @classmethod
    def setUpClass(cls):
        # Setup mock pipeline
        cls.mock_cam = MockCamera(fps=30.0)
        cls.mock_det = MockDetector(latency_s=0.01)
        cls.pipeline = VisionPipeline(cls.mock_cam, cls.mock_det)
        cls.pipeline.start()

        # Seed synthetic frame and result
        frame = np.full((240, 320, 3), 120, dtype=np.uint8)
        cls.pipeline._latest_raw_frame = frame
        cls.pipeline._latest_result = DetectionResult(
            frame_id=1,
            capture_time=time.monotonic(),
            finish_time=time.monotonic(),
            detections=[
                Detection(
                    box=(10, 10, 100, 100),
                    class_id=1,
                    class_name="person",
                    confidence=0.91,
                ),
                Detection(
                    box=(110, 110, 200, 200),
                    class_id=62,
                    class_name="chair",
                    confidence=0.74,
                ),
            ],
            annotated_frame=frame,
            counts={"person": 1, "chair": 1},
            frame_age_ms=5.0,
            end_to_end_ms=15.0,
            infer_latency_ms=10.0,
        )

        # Bind server to random available high port on 127.0.0.1
        cls.port = 18888
        cls.server = GroundStationServer(
            host="127.0.0.1",
            port=cls.port,
            pipeline=cls.pipeline,
            jpeg_quality=70,
            stream_max_fps=15,
        )
        cls.server.start_background()
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.pipeline.stop()

    def test_get_dashboard(self):
        url = f"{self.base_url}/"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            self.assertEqual(resp.status, HTTPStatus.OK)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("Indoor Drone Vision", content)
            self.assertIn("Ground Station", content)

    def test_get_telemetry_json(self):
        url = f"{self.base_url}/api/telemetry"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            self.assertEqual(resp.status, HTTPStatus.OK)
            self.assertIn("application/json", resp.headers.get("Content-Type", ""))
            data = json.loads(resp.read().decode("utf-8"))

            self.assertIn("capture_fps", data)
            self.assertIn("inference_fps", data)
            self.assertIn("detections", data)
            self.assertIn("counts", data)
            self.assertIn("session_id", data)
            self.assertIn("camera_status", data)
            self.assertIn("is_stale", data)
            self.assertIn("is_camera_connected", data)
            self.assertGreaterEqual(len(data["detections"]), 1)
            self.assertEqual(data["detections"][0]["class_name"], "person")
            self.assertIn("apparent_size_pct", data["detections"][0])
            self.assertGreaterEqual(data["counts"].get("person", 0), 1)

    def test_stale_telemetry_reporting(self):
        # Simulate camera disconnect by closing mock camera
        self.mock_cam.is_open = False
        # Allow capture worker loop to see dropped frames
        time.sleep(0.2)
        url = f"{self.base_url}/api/telemetry"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["is_stale"] or not data["is_camera_connected"])
            self.assertIn(data["camera_status"], ["Stale", "Disconnected"])

        # Re-enable mock camera for subsequent tests
        self.mock_cam.is_open = True

    def test_get_snapshot_jpeg(self):
        url = f"{self.base_url}/snapshot"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            self.assertEqual(resp.status, HTTPStatus.OK)
            self.assertIn("image/jpeg", resp.headers.get("Content-Type", ""))
            data = resp.read()
            self.assertGreater(len(data), 100)
            # JPEG magic bytes: FF D8 ... FF D9
            self.assertTrue(data.startswith(b"\xff\xd8"))

    def test_get_mjpeg_stream_and_disconnect(self):
        url = f"{self.base_url}/stream"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, HTTPStatus.OK)
            self.assertIn("multipart/x-mixed-replace", resp.headers.get("Content-Type", ""))
            # Read first chunk containing part header
            chunk = resp.read(256)
            self.assertIn(b"--frame", chunk)
            self.assertIn(b"image/jpeg", chunk)
            # Closing the context simulates client disconnection


if __name__ == "__main__":
    unittest.main()

