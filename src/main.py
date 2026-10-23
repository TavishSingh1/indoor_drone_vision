"""
main.py — Entry point for the indoor drone vision system with object detection
and ground-station streaming.

Features:
- Decoupled real-time vision pipeline: Camera capture runs at native 30 FPS
  while inference processes freshest available frame without lag backlog.
- Lightweight Ground-Station web server providing live telemetry and MJPEG stream.
- Headless test modes and GUI display mode.

Safety: Observational vision only. No connection to flight control or actuators.

Usage:
    # Phase 1 camera-only test:
    python -m src.main --headless --test

    # Phase 2 synchronous detection test:
    python -m src.main --headless --test --detect

    # Phase 3 decoupled real-time detection & telemetry:
    python -m src.main --detect --headless

    # Phase 3 with Ground Station web stream (binds to localhost by default):
    python -m src.main --detect --headless --stream

    # LAN streaming for Wi-Fi ground station (requires trusted private network):
    python -m src.main --detect --headless --stream --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import yaml

from src.camera import Camera, CameraConfig, CameraError, save_frame_jpeg
from src.detector import (
    Detection,
    DetectorConfig,
    InferenceError,
    ModelLoadError,
    ObjectDetector,
)
from src.pipeline import VisionPipeline
from src.server import GroundStationServer
from src.session_logger import SessionConfig, SessionLogger

# --------------------------------------------------------------------------- #
#  Constants                                                                  #
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Logging setup                                                              #
# --------------------------------------------------------------------------- #

def _setup_logging(level_name: str = "INFO", verbose: bool = False) -> None:
    if verbose:
        level = logging.DEBUG
    else:
        level = getattr(logging, level_name.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
#  Config loader                                                              #
# --------------------------------------------------------------------------- #

def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Load the YAML configuration file."""
    if not path.exists():
        logger.warning("Config file %s not found — using defaults", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def camera_config_from_dict(d: dict) -> CameraConfig:
    """Build a CameraConfig from config dict."""
    cam = d.get("camera", {})
    return CameraConfig(
        device=cam.get("device", "/dev/video0"),
        width=cam.get("width", 640),
        height=cam.get("height", 480),
        fourcc=cam.get("fourcc", "MJPG"),
        fps=cam.get("fps", 30),
        backend=cam.get("backend", cv2.CAP_V4L2),
    )


def detector_config_from_dict(d: dict) -> DetectorConfig:
    """Build a DetectorConfig from config dict."""
    det = d.get("detector", {})
    model_path = det.get("model_path", "models/ssd_mobilenet_v1_int8.onnx")
    resolved_model = str(PROJECT_ROOT / model_path) if not Path(model_path).is_absolute() else model_path

    return DetectorConfig(
        model_path=resolved_model,
        input_width=det.get("input_width", 300),
        input_height=det.get("input_height", 300),
        confidence_threshold=det.get("confidence_threshold", 0.50),
        nms_threshold=det.get("nms_threshold", 0.45),
        num_threads=det.get("num_threads", 2),
    )


# --------------------------------------------------------------------------- #
#  Display helper                                                             #
# --------------------------------------------------------------------------- #

def _has_display() -> bool:
    """Check whether a GUI display is available (X11 / Wayland)."""
    import os

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# --------------------------------------------------------------------------- #
#  Execution modes                                                            #
# --------------------------------------------------------------------------- #

def run_headless_test(
    camera: Camera,
    cfg: dict,
    detector: Optional[ObjectDetector] = None,
    session_logger: Optional[SessionLogger] = None,
) -> bool:
    """
    Capture a fixed number of frames, optionally run detection, save JPEGs, and exit.
    Preserves exact Phase 1 & Phase 2 verification behavior.
    """
    test_logger = logging.getLogger("headless_test")
    capture_cfg = cfg.get("capture", {})
    n_frames = capture_cfg.get("test_frame_count", 10)
    quality = capture_cfg.get("jpeg_quality", 90)

    output_dir = PROJECT_ROOT / capture_cfg.get("output_dir", "output")
    snapshot_dir = PROJECT_ROOT / capture_cfg.get("snapshot_dir", "output/snapshots")
    target_dir = snapshot_dir if detector is not None else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    mode_desc = "with detection" if detector is not None else "camera-only"
    test_logger.info(
        "Headless test (%s): capturing %d frames to %s",
        mode_desc,
        n_frames,
        target_dir,
    )

    if session_logger is not None:
        session_logger.start()

    success_count = 0
    total_latency_s = 0.0
    inference_count = 0

    for i in range(n_frames):
        frame = camera.read()
        if frame is None:
            test_logger.error("Frame %d/%d: capture failed", i + 1, n_frames)
            continue

        annotated_frame = frame
        if detector is not None:
            try:
                detections, latency = detector.infer(frame)
                total_latency_s += latency
                inference_count += 1
                counts = ObjectDetector.count_by_class(detections)
                count_str = ", ".join(f"{k}:{v}" for k, v in counts.items()) or "none"
                test_logger.info(
                    "Frame %d/%d: %d detections (%s) [infer: %.1f ms]",
                    i + 1,
                    n_frames,
                    len(detections),
                    count_str,
                    latency * 1000,
                )
                annotated_frame = ObjectDetector.annotate(
                    frame,
                    detections,
                    latency_ms=latency * 1000,
                    capture_fps=camera.capture_fps,
                )
                if session_logger is not None and detections:
                    session_logger.log_detections(frame_id=i + 1, detections=detections)
            except InferenceError as e:
                test_logger.error("Frame %d: inference failed: %s", i + 1, e)

        prefix = "detected_frame" if detector is not None else "test_frame"
        path = target_dir / f"{prefix}_{i:04d}.jpg"
        if save_frame_jpeg(annotated_frame, path, quality=quality):
            success_count += 1
            test_logger.debug("Saved frame to %s", path)
        else:
            test_logger.error("Frame %d/%d: save failed", i + 1, n_frames)

    cam_fps = camera.capture_fps
    mean_lat_ms = (total_latency_s / inference_count * 1000) if inference_count > 0 else 0.0
    infer_fps = (inference_count / total_latency_s) if total_latency_s > 0 else 0.0

    test_logger.info(
        "Headless test complete: %d/%d frames saved. Camera: %.1f FPS | Mean infer latency: %.1f ms (%.1f infer FPS)",
        success_count,
        n_frames,
        cam_fps,
        mean_lat_ms,
        infer_fps,
    )
    if session_logger is not None:
        from src.pipeline import PipelineTelemetry, get_system_telemetry

        temp_c, throttled_hex, mem_mb = get_system_telemetry()
        test_telem = PipelineTelemetry(
            capture_fps=cam_fps,
            inference_fps=infer_fps,
            inference_latency_ms=mean_lat_ms,
            frame_age_ms=0.0,
            total_captured=n_frames,
            total_inferred=inference_count,
            dropped_frames=0,
            temperature_c=temp_c,
            throttled_hex=throttled_hex,
            memory_available_mb=mem_mb,
            is_camera_connected=True,
            is_stale=False,
        )
        session_logger.log_telemetry_sample(test_telem)
        session_logger.close()

    return success_count == n_frames


def run_pipeline_continuous(
    pipeline: VisionPipeline,
    cfg: dict,
    headless: bool,
    server: Optional[GroundStationServer] = None,
    session_logger: Optional[SessionLogger] = None,
) -> None:
    """
    Phase 3/4 real-time loop using decoupled VisionPipeline, GroundStationServer,
    and SessionLogger.
    """
    loop_logger = logging.getLogger("pipeline_loop")
    display_cfg = cfg.get("display", {})
    capture_cfg = cfg.get("capture", {})
    fps_interval = capture_cfg.get("fps_report_interval", 5.0)
    window_name = display_cfg.get("window_name", "Indoor Drone Vision")

    show_gui = (not headless) and _has_display()
    if show_gui:
        loop_logger.info("Display mode: GUI window '%s'", window_name)
    else:
        loop_logger.info("Display mode: Headless (streaming/telemetry active)")

    running = True

    def _signal_handler(sig, _frame):
        nonlocal running
        loop_logger.info("Received shutdown signal %s — stopping gracefully...", sig)
        running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if session_logger is not None:
        session_logger.start()

    pipeline.start()
    if server is not None:
        server.start_background()

    last_report_time = time.monotonic()

    try:
        while running:
            # GUI window update if enabled
            if show_gui:
                display_frame = pipeline.get_latest_frame(annotated=True)
                if display_frame is not None:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(15) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' or ESC
                        loop_logger.info("Quit requested via GUI key")
                        break
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.05)

            # Periodic telemetry log report
            now = time.monotonic()
            if now - last_report_time >= fps_interval:
                telem = pipeline.get_telemetry()
                res = pipeline.get_latest_result()
                summary = "none"
                if res and res.counts:
                    summary = ", ".join(f"{k}:{v}" for k, v in sorted(res.counts.items()))

                temp_str = f"{telem.temperature_c:.1f}°C" if telem.temperature_c else "N/A"
                mem_str = f"{telem.memory_available_mb:.0f} MB" if telem.memory_available_mb else "N/A"
                status_str = "STALE" if telem.is_stale else "OK"

                loop_logger.info(
                    "Cam: %.1f FPS [%s] | Infer: %.1f FPS (%.0f ms, age: %.0f ms) | Temp: %s | Free RAM: %s | Objects: %s",
                    telem.capture_fps,
                    status_str,
                    telem.inference_fps,
                    telem.inference_latency_ms,
                    telem.frame_age_ms,
                    temp_str,
                    mem_str,
                    summary,
                )
                last_report_time = now

    finally:
        if server is not None:
            server.stop()
        pipeline.stop()
        if session_logger is not None:
            session_logger.close()
        if show_gui:
            cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Indoor Drone Vision — Camera capture, Object Detection & Ground-Station Streaming",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        default=False,
        help="Enable object detection model pipeline",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Enable local ground-station web dashboard and MJPEG stream",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Ground-station web server bind address (e.g. 127.0.0.1 or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Ground-station web server TCP port",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without graphical display",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run fixed frame count test and save images (Phase 1/2 mode)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        default=False,
        help="Force synchronous capture-infer loop (Phase 2 legacy mode)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Application main entry point."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    log_level = cfg.get("logging", {}).get("level", "INFO")
    _setup_logging(level_name=log_level, verbose=args.verbose)

    cam_cfg = camera_config_from_dict(cfg)
    det_cfg = detector_config_from_dict(cfg)

    logger.info("Initializing Indoor Drone Vision...")
    logger.info(
        "Camera device: %s (%dx%d %s @%d fps)",
        cam_cfg.device,
        cam_cfg.width,
        cam_cfg.height,
        cam_cfg.fourcc,
        cam_cfg.fps,
    )

    # Initialize detector if requested
    detector: Optional[ObjectDetector] = None
    if args.detect:
        logger.info("Object detection requested. Target model: %s", det_cfg.model_path)
        detector = ObjectDetector(det_cfg)
        try:
            detector.load_model()
        except ModelLoadError as e:
            logger.error("Detector initialization failed: %s", e)
            return 1

    # Camera instance
    camera = Camera(cam_cfg)

    # Session logger initialization
    sess_cfg = cfg.get("session_logger", {})
    session_logger: Optional[SessionLogger] = None
    if sess_cfg.get("enabled", True):
        session_logger = SessionLogger(
            config=SessionConfig(
                enabled=sess_cfg.get("enabled", True),
                output_dir=str(PROJECT_ROOT / sess_cfg.get("output_dir", "output/sessions")),
                flush_interval_s=sess_cfg.get("flush_interval_s", 2.0),
                queue_maxsize=sess_cfg.get("queue_maxsize", 1000),
                save_detection_csv=sess_cfg.get("save_detection_csv", True),
                save_summary_json=sess_cfg.get("save_summary_json", True),
            ),
            model_name="MobileNet-SSD v1" if detector else "None",
            model_path=det_cfg.model_path if detector else "None",
            app_config=cfg,
        )

    # If --test is requested, run fixed frame test
    if args.test:
        try:
            camera.open()
            ok = run_headless_test(camera, cfg, detector=detector, session_logger=session_logger)
            return 0 if ok else 1
        except CameraError as e:
            logger.error("Camera open failed: %s", e)
            return 1
        finally:
            camera.release()
            logger.info("Test execution complete.")

    # Determine server settings
    server_cfg = cfg.get("server", {})
    start_server = args.stream or server_cfg.get("enabled", False)
    server_host = args.host or server_cfg.get("host", "127.0.0.1")
    server_port = args.port or server_cfg.get("port", 8080)

    # Phase 3 Real-time Decoupled Pipeline
    buffer_size = cfg.get("pipeline", {}).get("max_buffer_size", 1)
    pipeline = VisionPipeline(
        camera=camera,
        detector=detector,
        max_buffer_size=buffer_size,
        session_logger=session_logger,
    )

    server: Optional[GroundStationServer] = None
    if start_server:
        server = GroundStationServer(
            host=server_host,
            port=server_port,
            pipeline=pipeline,
            jpeg_quality=server_cfg.get("jpeg_quality", 80),
            stream_max_fps=server_cfg.get("stream_max_fps", 25),
        )

    headless = args.headless or cfg.get("display", {}).get("headless", False)

    try:
        run_pipeline_continuous(
            pipeline=pipeline,
            cfg=cfg,
            headless=headless,
            server=server,
            session_logger=session_logger,
        )
        return 0
    except Exception:
        logger.exception("Unexpected error in vision pipeline")
        return 1
    finally:
        logger.info("System shutdown complete.")


if __name__ == "__main__":
    sys.exit(main())
