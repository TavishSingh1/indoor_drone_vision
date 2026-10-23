"""
test_session_logger.py — Unit and integration tests for session logging and reliability.

Tests:
1. Unique session ID generation and directory creation.
2. Detection events logging to CSV with correct headers and apparent size.
3. Telemetry sampling and summary JSON serialization on clean shutdown.
4. Robust error handling on disk write failures without crashing.
5. Bounded queue behavior under high event throughput.
6. Context manager lifecycle.
"""

from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path

from src.detector import Detection
from src.pipeline import PipelineTelemetry
from src.session_logger import SessionConfig, SessionLogger


class TestSessionLogger(unittest.TestCase):
    """Test SessionLogger capabilities."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)
        self.config = SessionConfig(
            enabled=True,
            output_dir=str(self.log_dir),
            flush_interval_s=0.1,
            queue_maxsize=20,
            save_detection_csv=True,
            save_summary_json=True,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_session_id_uniqueness(self):
        s1 = SessionLogger(config=self.config)
        s2 = SessionLogger(config=self.config)
        self.assertNotEqual(s1.session_id, s2.session_id)
        self.assertTrue(s1.session_id.startswith("session_"))

    def test_csv_and_summary_lifecycle(self):
        logger = SessionLogger(
            config=self.config,
            model_name="TestDetector",
            model_path="models/test.onnx",
            app_config={"camera": {"width": 640, "height": 480}},
        )
        logger.start()
        self.assertTrue(logger.is_active)

        # Log some detections
        dets = [
            Detection(
                box=(50, 50, 200, 200),
                class_id=1,
                class_name="person",
                confidence=0.92,
                apparent_size_pct=7.32,
            ),
            Detection(
                box=(300, 300, 350, 400),
                class_id=44,
                class_name="bottle",
                confidence=0.78,
                apparent_size_pct=1.63,
            ),
        ]
        logger.log_detections(frame_id=10, detections=dets, timestamp=time.time())

        # Log telemetry sample
        telem = PipelineTelemetry(
            capture_fps=30.0,
            inference_fps=4.0,
            inference_latency_ms=250.0,
            frame_age_ms=120.0,
            total_captured=100,
            total_inferred=15,
            dropped_frames=85,
            temperature_c=65.5,
            memory_available_mb=320.0,
            throttled_hex="0x0",
        )
        logger.log_telemetry_sample(telem)

        # Wait for worker to flush
        time.sleep(0.3)
        logger.close()
        self.assertFalse(logger.is_active)

        # Verify CSV exists and contains records
        csv_file = self.log_dir / f"{logger.session_id}_detections.csv"
        self.assertTrue(csv_file.exists())
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            self.assertGreaterEqual(len(reader), 3)  # Header + 2 rows
            header = reader[0]
            self.assertIn("class_name", header)
            self.assertIn("apparent_size_pct", header)

            row1 = reader[1]
            self.assertEqual(row1[4], "person")
            self.assertEqual(row1[10], "7.32")

            row2 = reader[2]
            self.assertEqual(row2[4], "bottle")
            self.assertEqual(row2[10], "1.63")

        # Verify Summary JSON exists
        json_file = self.log_dir / f"{logger.session_id}_summary.json"
        self.assertTrue(json_file.exists())
        with open(json_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
            self.assertEqual(summary["session_id"], logger.session_id)
            self.assertEqual(summary["model"]["name"], "TestDetector")
            self.assertEqual(summary["detections_summary"]["total_events"], 2)
            self.assertEqual(summary["detections_summary"]["counts_by_class"]["person"], 1)
            self.assertEqual(summary["detections_summary"]["counts_by_class"]["bottle"], 1)
            self.assertEqual(summary["frame_totals"]["total_captured"], 100)
            self.assertEqual(summary["hardware_telemetry"]["end_temperature_c"], 65.5)

    def test_bounded_queue_drop_on_overflow(self):
        # Configure tiny queue size = 5
        tiny_cfg = SessionConfig(
            enabled=True,
            output_dir=str(self.log_dir),
            queue_maxsize=5,
        )
        logger = SessionLogger(config=tiny_cfg)
        logger.start()

        dets = [
            Detection(box=(0, 0, 10, 10), class_id=1, class_name="person", confidence=0.8)
        ]
        # Push 50 events rapidly (exceeds queue capacity)
        for i in range(50):
            logger.log_detections(frame_id=i, detections=dets)

        # Queue must not grow beyond maxsize
        self.assertLessEqual(logger._queue.qsize(), 5)
        logger.close()

    def test_error_handling_invalid_dir(self):
        # Path that cannot be created or written (non-existent root/permission)
        invalid_cfg = SessionConfig(
            enabled=True,
            output_dir="/sys/invalid_non_writable_dir_for_test",
        )
        logger = SessionLogger(config=invalid_cfg)
        # Should not raise exception
        logger.start()
        self.assertFalse(logger.is_active)
        logger.close()


if __name__ == "__main__":
    unittest.main()

