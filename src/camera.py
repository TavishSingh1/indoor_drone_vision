"""
camera.py — USB camera capture module for indoor drone vision.

Opens a V4L2 USB camera, requests MJPG at the configured resolution,
verifies the negotiated format, and provides a clean frame-reading
interface with proper resource cleanup.

Hardware target: Mi USB Webcam HD on /dev/video0 (Raspberry Pi 4).
Safety: This module accesses only the camera. It does NOT interact with
flight controllers, MAVLink, serial ports, motors, or RC controls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera configuration loaded from config.yaml."""

    device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fourcc: str = "MJPG"
    fps: int = 30
    backend: int = cv2.CAP_V4L2


@dataclass
class CameraInfo:
    """Read-back of what the camera actually negotiated."""

    actual_width: int = 0
    actual_height: int = 0
    actual_fps: float = 0.0
    actual_fourcc: str = ""
    backend_name: str = ""


class CameraError(Exception):
    """Raised when the camera cannot be opened or configured."""


class Camera:
    """
    USB camera capture via OpenCV V4L2 backend.

    Usage::

        cam = Camera(config)
        cam.open()
        try:
            while True:
                frame = cam.read()
                if frame is not None:
                    # process frame ...
        finally:
            cam.release()

    Or as a context manager::

        with Camera(config) as cam:
            frame = cam.read()
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._info: Optional[CameraInfo] = None
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._is_open: bool = False

    # ------------------------------------------------------------------ #
    #  Properties                                                        #
    # ------------------------------------------------------------------ #

    @property
    def is_open(self) -> bool:
        """True if the camera is currently open and readable."""
        return self._is_open and self._cap is not None and self._cap.isOpened()

    @property
    def info(self) -> Optional[CameraInfo]:
        """Camera info populated after open()."""
        return self._info

    @property
    def frame_count(self) -> int:
        """Total number of frames read since open()."""
        return self._frame_count

    @property
    def capture_fps(self) -> float:
        """Measured capture FPS (frames read / elapsed time)."""
        if self._frame_count == 0 or self._start_time == 0.0:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                         #
    # ------------------------------------------------------------------ #

    def open(self) -> CameraInfo:
        """
        Open the camera device and configure the requested format.

        Returns the negotiated :class:`CameraInfo`.
        Raises :class:`CameraError` if the device cannot be opened.
        """
        if self._is_open:
            logger.warning("Camera already open — skipping re-open")
            return self._info  # type: ignore[return-value]

        device = self._config.device
        backend = self._config.backend

        logger.info("Opening camera: device=%s backend=%s", device, backend)

        # Verify device exists before attempting open
        dev_path = Path(device)
        if not dev_path.exists():
            raise CameraError(f"Camera device {device} does not exist")

        cap = cv2.VideoCapture(device, backend)
        if not cap.isOpened():
            raise CameraError(
                f"Failed to open camera {device} with backend {backend}"
            )

        # Request format — the camera may negotiate a different resolution
        fourcc_code = cv2.VideoWriter.fourcc(*self._config.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        # Read back negotiated values
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        raw_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = _decode_fourcc(raw_fourcc)
        backend_name = cap.getBackendName()

        self._info = CameraInfo(
            actual_width=actual_w,
            actual_height=actual_h,
            actual_fps=actual_fps,
            actual_fourcc=actual_fourcc,
            backend_name=backend_name,
        )

        # Log negotiation results
        req = f"{self._config.width}x{self._config.height} {self._config.fourcc} @{self._config.fps}fps"
        got = f"{actual_w}x{actual_h} {actual_fourcc} @{actual_fps:.1f}fps"
        if (
            actual_w != self._config.width
            or actual_h != self._config.height
            or actual_fourcc != self._config.fourcc
        ):
            logger.warning("Format mismatch — requested %s, got %s", req, got)
        else:
            logger.info("Camera configured: %s (backend: %s)", got, backend_name)

        # Warm up — discard the first frame (often partially exposed)
        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise CameraError("Camera opened but failed to read initial frame")

        self._cap = cap
        self._is_open = True
        self._frame_count = 0
        self._start_time = time.monotonic()

        logger.info("Camera ready")
        return self._info

    def read(self) -> Optional[np.ndarray]:
        """
        Read one frame from the camera.

        Returns the BGR numpy array, or ``None`` if the read failed.
        """
        if not self.is_open:
            logger.error("Attempted to read from a closed camera")
            return None

        ret, frame = self._cap.read()  # type: ignore[union-attr]
        if not ret or frame is None:
            logger.warning("Frame read failed (frame #%d)", self._frame_count)
            return None

        self._frame_count += 1
        return frame

    def release(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            logger.info(
                "Releasing camera (captured %d frames, avg %.1f fps)",
                self._frame_count,
                self.capture_fps,
            )
            try:
                self._cap.release()
            except Exception:
                logger.exception("Error releasing camera")
            finally:
                self._cap = None
                self._is_open = False

    # ------------------------------------------------------------------ #
    #  Context manager                                                   #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ------------------------------------------------------------------ #
    #  String representation                                             #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        return f"<Camera device={self._config.device!r} status={status}>"


# ---------------------------------------------------------------------- #
#  Helpers                                                               #
# ---------------------------------------------------------------------- #

def _decode_fourcc(v: int) -> str:
    """Decode an integer fourcc into a 4-character string."""
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def save_frame_jpeg(
    frame: np.ndarray,
    path: Path | str,
    quality: int = 90,
) -> bool:
    """
    Save a BGR frame as JPEG.

    Returns True on success, False on failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[0]
        if ok:
            ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            logger.debug("Saved frame to %s", path)
        else:
            logger.error("Failed to encode/write JPEG to %s", path)
        return bool(ok)
    except Exception:
        logger.exception("Error saving frame to %s", path)
        return False

