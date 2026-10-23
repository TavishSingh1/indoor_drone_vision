"""
pipeline.py — Thread-safe decoupled real-time vision pipeline.

Decouples camera capture from neural network inference:
1. CaptureWorker: Reads camera at native hardware rate (~30 FPS).
2. LatestFrameBuffer: Bounded buffer (capacity=1) with stale-frame replacement.
3. InferenceWorker: Runs object detection on the freshest available frame.
4. LatestResultBuffer: Stores the most recent annotated frame and detection metadata.
5. Telemetry: Tracks capture FPS, inference FPS, inference latency, frame age, and end-to-end latency.

Safety: Observational vision only. No connection to flight control or actuators.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.camera import Camera, CameraConfig, CameraError
from src.detector import Detection, ObjectDetector

logger = logging.getLogger(__name__)


@dataclass
class FramePacket:
    """A captured frame tagged with sequence ID and high-resolution timestamp."""

    frame_id: int
    capture_time: float
    frame: np.ndarray


@dataclass
class PipelineTelemetry:
    """Real-time performance and latency telemetry."""

    capture_fps: float = 0.0
    inference_fps: float = 0.0
    inference_latency_ms: float = 0.0
    frame_age_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    total_captured: int = 0
    total_inferred: int = 0
    dropped_frames: int = 0
    temperature_c: Optional[float] = None
    throttled_hex: Optional[str] = None
    memory_available_mb: Optional[float] = None
    is_camera_connected: bool = True
    is_stale: bool = False


@dataclass
class DetectionResult:
    """Latest detection outputs and annotated visualization."""

    frame_id: int
    capture_time: float
    finish_time: float
    detections: list[Detection] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None
    counts: dict[str, int] = field(default_factory=dict)
    frame_age_ms: float = 0.0
    end_to_end_ms: float = 0.0
    infer_latency_ms: float = 0.0


class LatestFrameBuffer:
    """
    Thread-safe bounded buffer (max capacity 1) with stale-frame dropping.

    When the camera captures faster than inference, the newest frame replaces
    the pending frame so inference never processes a stale backlog.
    """

    def __init__(self, maxsize: int = 1) -> None:
        self._maxsize = max(1, maxsize)
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._packet: Optional[FramePacket] = None
        self._dropped_count: int = 0

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def put(self, packet: FramePacket) -> None:
        """Store newest frame, discarding any older unprocessed frame."""
        with self._lock:
            if self._packet is not None:
                self._dropped_count += 1
            self._packet = packet
            self._new_frame_event.set()

    def get(self, timeout: float = 0.1) -> Optional[FramePacket]:
        """
        Wait for and retrieve the newest frame.

        Clears the buffer so the same frame is not processed twice.
        """
        if not self._new_frame_event.wait(timeout=timeout):
            return None

        with self._lock:
            packet = self._packet
            self._packet = None
            self._new_frame_event.clear()
            return packet

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._packet = None
            self._new_frame_event.clear()


def get_system_telemetry() -> tuple[Optional[float], Optional[str], Optional[float]]:
    """Query Raspberry Pi SoC temperature, throttling flags, and available memory."""
    temp_c: Optional[float] = None
    throttled_hex: Optional[str] = None
    mem_avail_mb: Optional[float] = None

    # Try thermal zone sysfs (fast, no subprocess)
    try:
        thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_path.exists():
            raw = thermal_path.read_text(encoding="utf-8").strip()
            temp_c = round(float(raw) / 1000.0, 1)
    except Exception:
        pass

    # Try /proc/meminfo for available RAM
    try:
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.exists():
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    mem_avail_mb = round(kb / 1024.0, 1)
                    break
    except Exception:
        pass

    # Try vcgencmd for throttling flags
    try:
        import subprocess
        res = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=0.5,
            check=False,
        )
        if res.returncode == 0 and "throttled=" in res.stdout:
            throttled_hex = res.stdout.strip().split("=")[-1]
    except Exception:
        pass

    return temp_c, throttled_hex, mem_avail_mb


class VisionPipeline:
    """
    Decoupled multi-threaded vision pipeline.

    - Thread 1 (CaptureWorker): Camera -> LatestFrameBuffer (~30 FPS)
    - Thread 2 (InferenceWorker): LatestFrameBuffer -> ObjectDetector -> LatestResultBuffer (~3-4 FPS)
    """

    def __init__(
        self,
        camera: Camera,
        detector: Optional[ObjectDetector] = None,
        max_buffer_size: int = 1,
        session_logger: Optional[Any] = None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.session_logger = session_logger
        self.frame_buffer = LatestFrameBuffer(maxsize=max_buffer_size)

        self._shutdown_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None

        # Thread-safe result storage
        self._result_lock = threading.Lock()
        self._latest_result: Optional[DetectionResult] = None
        self._latest_raw_frame: Optional[np.ndarray] = None

        # Metrics tracking
        self._metrics_lock = threading.Lock()
        self._capture_count = 0
        self._inference_count = 0
        self._capture_fps = 0.0
        self._inference_fps = 0.0
        self._last_infer_latency_ms = 0.0
        self._last_frame_age_ms = 0.0
        self._last_end_to_end_ms = 0.0
        self._last_capture_time = 0.0
        self._is_camera_connected = True
        self._consecutive_cam_failures = 0

        # FPS calculation windows
        self._last_cap_time = time.monotonic()
        self._last_cap_frames = 0
        self._last_inf_time = time.monotonic()
        self._last_inf_frames = 0

    @property
    def is_running(self) -> bool:
        return not self._shutdown_event.is_set() and (
            (self._capture_thread is not None and self._capture_thread.is_alive())
        )

    def start(self) -> None:
        """Start the decoupled capture and inference workers."""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return

        self._shutdown_event.clear()
        self.frame_buffer.clear()

        # Ensure camera is open
        if not self.camera.is_open:
            self.camera.open()

        # Start capture worker
        self._capture_thread = threading.Thread(
            target=self._capture_worker_loop,
            name="CaptureWorker",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info("CaptureWorker thread started")

        # Start inference worker if detector provided
        if self.detector is not None:
            if not self.detector.is_loaded:
                self.detector.load_model()
            self._inference_thread = threading.Thread(
                target=self._inference_worker_loop,
                name="InferenceWorker",
                daemon=True,
            )
            self._inference_thread.start()
            logger.info("InferenceWorker thread started")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal workers to stop and wait for completion."""
        logger.info("Stopping vision pipeline...")
        self._shutdown_event.set()

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
            self._capture_thread = None

        if self._inference_thread is not None and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=timeout)
            self._inference_thread = None

        # Cleanly release camera hardware
        self.camera.release()
        self.frame_buffer.clear()
        logger.info("Vision pipeline stopped cleanly")

    def __enter__(self) -> "VisionPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    #  Worker Loops                                                      #
    # ------------------------------------------------------------------ #

    def _capture_worker_loop(self) -> None:
        """Continuously grab frames from camera at maximum hardware rate."""
        frame_seq = 0
        window_start = time.monotonic()
        window_frames = 0

        while not self._shutdown_event.is_set():
            t_cap = time.monotonic()
            frame = self.camera.read()
            if frame is None:
                # Camera read dropped or camera disconnected
                self._consecutive_cam_failures += 1
                if self._consecutive_cam_failures >= 15:
                    self._is_camera_connected = False
                time.sleep(0.005)
                continue

            self._consecutive_cam_failures = 0
            self._is_camera_connected = True
            self._last_capture_time = t_cap

            frame_seq += 1
            window_frames += 1
            with self._metrics_lock:
                self._capture_count = frame_seq

            # Store latest raw frame for server or display
            with self._result_lock:
                self._latest_raw_frame = frame

            # Push to bounded buffer for inference
            packet = FramePacket(
                frame_id=frame_seq,
                capture_time=t_cap,
                frame=frame,
            )
            self.frame_buffer.put(packet)

            # Update capture FPS metrics every 1 second
            now = time.monotonic()
            elapsed = now - window_start
            if elapsed >= 1.0:
                with self._metrics_lock:
                    self._capture_fps = window_frames / elapsed
                window_frames = 0
                window_start = now

    def _inference_worker_loop(self) -> None:
        """Continuously pull newest frame and run model inference."""
        assert self.detector is not None

        window_start = time.monotonic()
        window_inferences = 0

        while not self._shutdown_event.is_set():
            # Wait for freshest frame (drop older frames automatically)
            packet = self.frame_buffer.get(timeout=0.05)
            if packet is None:
                continue

            t_infer_start = time.monotonic()
            frame_age_ms = (t_infer_start - packet.capture_time) * 1000.0

            try:
                detections, infer_latency = self.detector.infer(packet.frame)
                t_finish = time.monotonic()
                end_to_end_ms = (t_finish - packet.capture_time) * 1000.0
                infer_latency_ms = infer_latency * 1000.0

                counts = ObjectDetector.count_by_class(detections)

                # Record detection events in session logger if enabled
                if self.session_logger is not None and detections:
                    try:
                        self.session_logger.log_detections(
                            frame_id=packet.frame_id,
                            detections=detections,
                            timestamp=packet.capture_time,
                        )
                    except Exception as e:
                        logger.debug("Session logging error: %s", e)

                # Generate annotated visualization
                with self._metrics_lock:
                    current_cap_fps = self._capture_fps
                    current_inf_fps = self._inference_fps

                annotated = ObjectDetector.annotate(
                    packet.frame,
                    detections,
                    inference_fps=current_inf_fps,
                    capture_fps=current_cap_fps,
                    latency_ms=infer_latency_ms,
                )

                result = DetectionResult(
                    frame_id=packet.frame_id,
                    capture_time=packet.capture_time,
                    finish_time=t_finish,
                    detections=detections,
                    annotated_frame=annotated,
                    counts=counts,
                    frame_age_ms=frame_age_ms,
                    end_to_end_ms=end_to_end_ms,
                    infer_latency_ms=infer_latency_ms,
                )

                with self._result_lock:
                    self._latest_result = result

                window_inferences += 1
                with self._metrics_lock:
                    self._inference_count += 1
                    self._last_infer_latency_ms = infer_latency_ms
                    self._last_frame_age_ms = frame_age_ms
                    self._last_end_to_end_ms = end_to_end_ms

            except Exception as e:
                logger.error("Inference worker error on frame %d: %s", packet.frame_id, e)

            # Update inference FPS metrics every 1 second
            now = time.monotonic()
            elapsed = now - window_start
            if elapsed >= 1.0:
                with self._metrics_lock:
                    self._inference_fps = window_inferences / elapsed
                window_inferences = 0
                window_start = now

    # ------------------------------------------------------------------ #
    #  Public Query API                                                  #
    # ------------------------------------------------------------------ #

    def get_latest_result(self) -> Optional[DetectionResult]:
        """Thread-safe getter for the latest detection result."""
        with self._result_lock:
            return self._latest_result

    def get_latest_frame(self, annotated: bool = True) -> Optional[np.ndarray]:
        """
        Thread-safe getter for the latest image.

        If annotated=True and detections are available, returns annotated frame.
        Otherwise returns the latest raw camera frame.
        """
        with self._result_lock:
            if annotated and self._latest_result is not None:
                return self._latest_result.annotated_frame
            return self._latest_raw_frame

    def get_telemetry(self) -> PipelineTelemetry:
        """Thread-safe getter for current performance metrics, temperatures, and staleness."""
        temp_c, throttled, mem_avail_mb = get_system_telemetry()
        now = time.monotonic()
        with self._metrics_lock:
            # Mark stale if camera has captured in the past but no new frame for > 2.0s
            is_stale = (
                (self._capture_count > 0 and (now - self._last_capture_time > 2.0))
                or (not self._is_camera_connected)
            )
            telem = PipelineTelemetry(
                capture_fps=self._capture_fps,
                inference_fps=self._inference_fps,
                inference_latency_ms=self._last_infer_latency_ms,
                frame_age_ms=self._last_frame_age_ms,
                end_to_end_latency_ms=self._last_end_to_end_ms,
                total_captured=self._capture_count,
                total_inferred=self._inference_count,
                dropped_frames=self.frame_buffer.dropped_count,
                temperature_c=temp_c,
                throttled_hex=throttled,
                memory_available_mb=mem_avail_mb,
                is_camera_connected=self._is_camera_connected,
                is_stale=is_stale,
            )

        if self.session_logger is not None:
            try:
                self.session_logger.log_telemetry_sample(telem)
            except Exception as e:
                logger.debug("Session telemetry logging error: %s", e)

        return telem

