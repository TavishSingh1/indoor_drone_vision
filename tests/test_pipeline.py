"""
test_pipeline.py — Unit and integration tests for the decoupled vision pipeline.

Tests:
1. LatestFrameBuffer capacity and stale-frame dropping.
2. Concurrent capture and inference with mock camera and detector.
3. Thread-safe start, stop, timeout, and exception recovery.
4. Frame-age and latency computation.
"""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.detector import Detection
from src.pipeline import (
    DetectionResult,
    FramePacket,
    LatestFrameBuffer,
    PipelineTelemetry,
    VisionPipeline,
    get_system_telemetry,
)


class MockCamera:
    """Mock camera that simulates high-rate frame generation."""

    def __init__(self, fps: float = 60.0) -> None:
        self.fps = fps
        self.is_open = True
        self._frame_count = 0
        self._released = False

    def open(self):
        self.is_open = True
        self._released = False
        return None

    def read(self) -> Optional[np.ndarray]:
        if not self.is_open or self._released:
            return None
        self._frame_count += 1
        time.sleep(1.0 / self.fps)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        self._released = True
        self.is_open = False

    @property
    def frame_count(self) -> int:
        return self._frame_count


class MockDetector:
    """Mock detector that simulates CPU-bound inference latency."""

    def __init__(self, latency_s: float = 0.05) -> None:
        self.latency_s = latency_s
        self.is_loaded = True
        self.infer_count = 0

    def load_model(self) -> None:
        self.is_loaded = True

    def infer(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        self.infer_count += 1
        time.sleep(self.latency_s)
        dummy_det = [
            Detection(
                box=(50, 50, 200, 200),
                class_id=1,
                class_name="person",
                confidence=0.88,
            )
        ]
        return dummy_det, self.latency_s


class TestLatestFrameBuffer(unittest.TestCase):
    """Test buffer bounding and stale-frame dropping."""

    def test_buffer_put_and_get(self):
        buf = LatestFrameBuffer(maxsize=1)
        pkt1 = FramePacket(frame_id=1, capture_time=100.0, frame=np.zeros((10, 10, 3), dtype=np.uint8))
        buf.put(pkt1)

        ret = buf.get(timeout=0.1)
        self.assertIsNotNone(ret)
        self.assertEqual(ret.frame_id, 1)

        # Buffer is now empty
        self.assertIsNone(buf.get(timeout=0.01))

    def test_stale_frame_dropping(self):
        buf = LatestFrameBuffer(maxsize=1)
        pkt1 = FramePacket(frame_id=1, capture_time=100.0, frame=np.zeros((10, 10, 3), dtype=np.uint8))
        pkt2 = FramePacket(frame_id=2, capture_time=100.1, frame=np.zeros((10, 10, 3), dtype=np.uint8))
        pkt3 = FramePacket(frame_id=3, capture_time=100.2, frame=np.zeros((10, 10, 3), dtype=np.uint8))

        buf.put(pkt1)
        buf.put(pkt2)
        buf.put(pkt3)

        # 2 frames should have been dropped
        self.assertEqual(buf.dropped_count, 2)

        # The retrieved frame must be the freshest (frame_id 3)
        ret = buf.get(timeout=0.1)
        self.assertIsNotNone(ret)
        self.assertEqual(ret.frame_id, 3)


class TestVisionPipelineDecoupled(unittest.TestCase):
    """Test concurrent decoupled workers and thread safety."""

    def test_concurrent_capture_and_inference(self):
        mock_cam = MockCamera(fps=50.0)  # Produces frame every ~20ms
        mock_det = MockDetector(latency_s=0.03)  # Fast mock inference

        pipeline = VisionPipeline(
            camera=mock_cam,
            detector=mock_det,
            max_buffer_size=1,
        )

        pipeline.start()
        self.assertTrue(pipeline.is_running)

        # Run pipeline for 0.6 seconds
        time.sleep(0.6)

        # Check telemetry
        telem = pipeline.get_telemetry()
        self.assertGreaterEqual(telem.total_captured, 5)
        self.assertGreaterEqual(telem.total_inferred, 1)
        # Because camera (50 FPS) > inference (~33 FPS), stale frames must have been dropped
        self.assertGreater(telem.dropped_frames, 0)

        # Check latest frame and detections
        latest_res = pipeline.get_latest_result()
        self.assertIsNotNone(latest_res)
        self.assertEqual(len(latest_res.detections), 1)
        self.assertEqual(latest_res.detections[0].class_name, "person")

        # Frame age must be small (freshest frame, not accumulating lag)
        self.assertLess(latest_res.frame_age_ms, 200.0)

        # Test clean stop
        pipeline.stop(timeout=1.0)
        self.assertFalse(pipeline.is_running)
        self.assertTrue(mock_cam._released)

    def test_pipeline_context_manager(self):
        mock_cam = MockCamera(fps=60.0)
        mock_det = MockDetector(latency_s=0.02)

        with VisionPipeline(camera=mock_cam, detector=mock_det) as pipeline:
            self.assertTrue(pipeline.is_running)
            time.sleep(0.1)

        self.assertFalse(pipeline.is_running)
        self.assertTrue(mock_cam._released)


class TestSystemTelemetry(unittest.TestCase):
    """Test thermal and throttling reading."""

    def test_telemetry_query(self):
        temp, throttled, mem = get_system_telemetry()
        # On Pi 4, temp should be a float or None
        if temp is not None:
            self.assertIsInstance(temp, float)
            self.assertGreater(temp, 20.0)
            self.assertLess(temp, 110.0)
        if mem is not None:
            self.assertIsInstance(mem, float)
            self.assertGreater(mem, 0.0)


if __name__ == "__main__":
    unittest.main()

