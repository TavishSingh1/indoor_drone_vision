"""
session_logger.py — Persistent, resource-efficient flight vision session logger.

Records:
1. Session metadata (start/stop timestamps, config, model details, app version) -> JSON
2. Individual detection events -> CSV
3. Performance telemetry aggregates (capture FPS, infer FPS, latency, frame age, CPU temp, memory)

Safety & Resource Efficiency:
- Uses a bounded background queue (maxsize=1000) so logging never blocks camera or inference.
- Handles full disk (ENOSPC), permission errors, and OS exceptions gracefully without crashing vision.
- Periodically flushes to disk and finalizes cleanly on shutdown.
- Never retains raw video footage or logs personal information by default.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from src.detector import Detection
from src.pipeline import PipelineTelemetry

logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"


@dataclass
class SessionConfig:
    """Session logger configuration."""

    enabled: bool = True
    output_dir: str = "output/sessions"
    flush_interval_s: float = 2.0
    queue_maxsize: int = 1000
    save_detection_csv: bool = True
    save_summary_json: bool = True


class SessionLogger:
    """
    Thread-safe asynchronous session logger.
    """

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        model_name: str = "MobileNet-SSD v1",
        model_path: str = "models/ssd_mobilenet_v1_int8.onnx",
        app_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = config or SessionConfig()
        self.model_name = model_name
        self.model_path = model_path
        self.app_config = app_config or {}

        # Generate unique session ID with timestamp and random short suffix
        ts_slug = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand_suffix = os.urandom(2).hex()
        self.session_id = f"session_{ts_slug}_{rand_suffix}"

        self.start_epoch = time.time()
        self.start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.end_epoch: Optional[float] = None
        self.end_iso: Optional[str] = None

        self._output_dir = Path(self.config.output_dir)
        self._csv_path = self._output_dir / f"{self.session_id}_detections.csv"
        self._json_path = self._output_dir / f"{self.session_id}_summary.json"

        # Bounded event queue
        self._queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue(
            maxsize=self.config.queue_maxsize
        )
        self._shutdown_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # State tracking
        self._csv_file = None
        self._csv_writer = None
        self._is_open = False
        self._disk_error_logged = False

        # Metrics accumulators
        self._lock = threading.Lock()
        self._total_detections_logged = 0
        self._class_counts: dict[str, int] = {}
        self._last_telemetry: Optional[PipelineTelemetry] = None
        self._telemetry_samples: list[dict[str, Any]] = []

    @property
    def is_active(self) -> bool:
        return self._is_open and not self._shutdown_event.is_set()

    def start(self) -> None:
        """Initialize session files and launch background worker."""
        if not self.config.enabled:
            logger.info("Session logging is disabled in configuration")
            return

        if self._is_open:
            return

        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)

            if self.config.save_detection_csv:
                # Open CSV file with header
                self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(
                    [
                        "timestamp_iso",
                        "timestamp_epoch",
                        "frame_id",
                        "class_id",
                        "class_name",
                        "confidence",
                        "box_x1",
                        "box_y1",
                        "box_x2",
                        "box_y2",
                        "apparent_size_pct",
                    ]
                )
                self._csv_file.flush()

            self._is_open = True
            self._shutdown_event.clear()

            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"SessionLogger-{self.session_id}",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("SessionLogger started. Session ID: %s (dir: %s)", self.session_id, self._output_dir)

        except (OSError, PermissionError) as e:
            logger.error("Failed to initialize session logger on %s: %s", self._output_dir, e)
            self._is_open = False

    def log_detections(
        self,
        frame_id: int,
        detections: list[Detection],
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Queue detection events for persistent CSV logging.
        Non-blocking: if queue is full, drops the event to protect real-time pipeline.
        """
        if not self.is_active or not detections:
            return

        t = timestamp or time.time()
        iso = datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).isoformat()

        for det in detections:
            with self._lock:
                self._total_detections_logged += 1
                self._class_counts[det.class_name] = (
                    self._class_counts.get(det.class_name, 0) + 1
                )

            item = {
                "type": "detection",
                "row": [
                    iso,
                    f"{t:.4f}",
                    frame_id,
                    det.class_id,
                    det.class_name,
                    f"{det.confidence:.4f}",
                    det.box[0],
                    det.box[1],
                    det.box[2],
                    det.box[3],
                    f"{det.apparent_size_pct:.2f}",
                ],
            }
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                if not self._disk_error_logged:
                    logger.warning("SessionLogger queue is full; dropping detection log events")
                    self._disk_error_logged = True

    def log_telemetry_sample(self, telemetry: PipelineTelemetry) -> None:
        """Record a periodic telemetry sample for session aggregation."""
        with self._lock:
            self._last_telemetry = telemetry
            # Keep at most 100 historical samples in memory
            if len(self._telemetry_samples) >= 100:
                self._telemetry_samples.pop(0)

            self._telemetry_samples.append(
                {
                    "time": time.time(),
                    "capture_fps": telemetry.capture_fps,
                    "inference_fps": telemetry.inference_fps,
                    "latency_ms": telemetry.inference_latency_ms,
                    "frame_age_ms": telemetry.frame_age_ms,
                    "temp_c": telemetry.temperature_c,
                    "mem_mb": telemetry.memory_available_mb,
                }
            )

    def _worker_loop(self) -> None:
        """Background thread writing queued events to disk and flushing periodically."""
        last_flush = time.monotonic()

        while not self._shutdown_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                item = None

            if item is not None and item.get("type") == "detection":
                row = item.get("row")
                if self._csv_writer is not None and row is not None:
                    try:
                        self._csv_writer.writerow(row)
                    except (OSError, PermissionError) as e:
                        if not self._disk_error_logged:
                            logger.error("Disk write error in SessionLogger: %s", e)
                            self._disk_error_logged = True

            # Periodic flush
            now = time.monotonic()
            if now - last_flush >= self.config.flush_interval_s:
                self._flush_safe()
                last_flush = now

        self._flush_safe()

    def _flush_safe(self) -> None:
        """Safely flush CSV buffer without raising."""
        if self._csv_file is not None and not self._csv_file.closed:
            try:
                self._csv_file.flush()
            except Exception as e:
                logger.debug("CSV flush exception: %s", e)

    def close(self) -> None:
        """Stop logger, drain remaining events, and write session summary JSON."""
        if not self._is_open:
            return

        logger.info("Closing SessionLogger for %s...", self.session_id)
        self.end_epoch = time.time()
        self.end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._shutdown_event.set()

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
            self._worker_thread = None

        # Close CSV file
        if self._csv_file is not None and not self._csv_file.closed:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception as e:
                logger.debug("Error closing CSV file: %s", e)
            finally:
                self._csv_file = None

        # Write summary JSON
        if self.config.save_summary_json:
            self._write_summary_json()

        self._is_open = False
        logger.info("SessionLogger closed. Summary: %s", self._json_path)

    def _write_summary_json(self) -> None:
        """Generate and save the final session summary JSON."""
        duration_s = (
            (self.end_epoch - self.start_epoch) if self.end_epoch else 0.0
        )

        with self._lock:
            total_dets = self._total_detections_logged
            counts = dict(self._class_counts)
            samples = list(self._telemetry_samples)
            last_telem = self._last_telemetry

        # Calculate averages from samples
        avg_cap_fps = (
            round(sum(s["capture_fps"] for s in samples) / len(samples), 1)
            if samples
            else 0.0
        )
        avg_inf_fps = (
            round(sum(s["inference_fps"] for s in samples) / len(samples), 1)
            if samples
            else 0.0
        )
        avg_lat = (
            round(sum(s["latency_ms"] for s in samples) / len(samples), 1)
            if samples
            else 0.0
        )
        avg_age = (
            round(sum(s["frame_age_ms"] for s in samples) / len(samples), 1)
            if samples
            else 0.0
        )

        summary_data = {
            "session_id": self.session_id,
            "app_version": APP_VERSION,
            "timestamps": {
                "start_utc": self.start_iso,
                "end_utc": self.end_iso,
                "duration_seconds": round(duration_s, 2),
            },
            "model": {
                "name": self.model_name,
                "path": self.model_path,
            },
            "configuration": {
                "camera_width": self.app_config.get("camera", {}).get("width", 640),
                "camera_height": self.app_config.get("camera", {}).get("height", 480),
                "confidence_threshold": self.app_config.get("detector", {}).get("confidence_threshold", 0.5),
                "num_threads": self.app_config.get("detector", {}).get("num_threads", 2),
            },
            "frame_totals": {
                "total_captured": last_telem.total_captured if last_telem else 0,
                "total_inferred": last_telem.total_inferred if last_telem else 0,
                "total_dropped": last_telem.dropped_frames if last_telem else 0,
            },
            "performance_averages": {
                "mean_capture_fps": avg_cap_fps,
                "mean_inference_fps": avg_inf_fps,
                "mean_inference_latency_ms": avg_lat,
                "mean_frame_age_ms": avg_age,
            },
            "hardware_telemetry": {
                "end_temperature_c": last_telem.temperature_c if last_telem else None,
                "end_memory_available_mb": last_telem.memory_available_mb if last_telem else None,
                "throttled_hex": last_telem.throttled_hex if last_telem else None,
            },
            "detections_summary": {
                "total_events": total_dets,
                "counts_by_class": counts,
            },
        }

        try:
            temp_json = self._output_dir / f"{self._json_path.name}.tmp"
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)
            temp_json.replace(self._json_path)
        except (OSError, PermissionError) as e:
            logger.error("Failed to write session summary JSON: %s", e)

    def __enter__(self) -> "SessionLogger":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

