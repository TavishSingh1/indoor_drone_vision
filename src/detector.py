"""
detector.py — Pretrained object detection engine using ONNX Runtime.

Integrates a COCO-trained detector (SSD MobileNet v1) optimized for ARM64 CPU.
Handles preprocessing, tensor inference, postprocessing, confidence filtering,
NMS, class counting, and visual annotation.

Safety: Observational vision only. No connection to flight control or actuators.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from src.labels import get_label

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A detected object bounding box and classification."""

    # Absolute pixel box in original image: (xmin, ymin, xmax, ymax)
    box: tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    apparent_size_pct: float = 0.0  # Bounding box area as percentage of image area


@dataclass
class DetectorConfig:
    """Object detector configuration parameters."""

    model_path: str = "models/ssd_mobilenet_v1.onnx"
    input_width: int = 300
    input_height: int = 300
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    num_threads: int = 2


class ModelLoadError(Exception):
    """Raised when the ONNX model cannot be loaded or initialized."""


class InferenceError(Exception):
    """Raised when tensor inference fails or produces invalid shapes."""


class ObjectDetector:
    """
    Object detection engine using ONNX Runtime CPUExecutionProvider.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: str = ""
        self._output_names: list[str] = []
        self._load_duration_s: float = 0.0

    @property
    def is_loaded(self) -> bool:
        """True if the model session is initialized."""
        return self._session is not None

    @property
    def load_duration_s(self) -> float:
        """Seconds spent loading the model session."""
        return self._load_duration_s

    def load_model(self) -> None:
        """
        Load the ONNX model session with optimized thread count.

        Raises ModelLoadError if file is missing or ONNX Runtime initialization fails.
        """
        model_file = Path(self.config.model_path)
        if not model_file.is_file():
            raise ModelLoadError(
                f"Model file not found: {self.config.model_path}. "
                "Ensure model is downloaded and path is correct."
            )

        logger.info(
            "Loading ONNX detection model from %s (threads=%d)...",
            model_file,
            self.config.num_threads,
        )

        t0 = time.perf_counter()
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.num_threads
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            session = ort.InferenceSession(
                str(model_file),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            inputs = session.get_inputs()
            if not inputs:
                raise ModelLoadError("Model has no inputs defined")
            self._input_name = inputs[0].name
            self._output_names = [out.name for out in session.get_outputs()]

            self._session = session
            self._load_duration_s = time.perf_counter() - t0
            logger.info(
                "Model loaded in %.2f s. Input: %s, Outputs: %s",
                self._load_duration_s,
                self._input_name,
                self._output_names,
            )
        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise
            raise ModelLoadError(f"Failed to load ONNX model: {e}") from e

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess camera BGR frame for SSD MobileNet v1 input:
        1. Resize to (input_width, input_height)
        2. Convert BGR to RGB
        3. Expand dimensions to (1, H, W, 3) as uint8
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")

        resized = cv2.resize(
            frame,
            (self.config.input_width, self.config.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)
        return tensor

    def postprocess(
        self,
        raw_outputs: list[np.ndarray],
        orig_width: int,
        orig_height: int,
    ) -> list[Detection]:
        """
        Decode raw outputs, apply confidence threshold and optional NMS,
        and scale bounding boxes to original frame dimensions.

        Expected output tensors for SSD MobileNet v1:
        - raw_outputs[0]: boxes, shape (1, N, 4), normalized [ymin, xmin, ymax, xmax]
        - raw_outputs[1]: classes, shape (1, N)
        - raw_outputs[2]: scores, shape (1, N)
        - raw_outputs[3]: num_detections, shape (1,)
        """
        if len(raw_outputs) < 4:
            raise InferenceError(
                f"Expected at least 4 output tensors, got {len(raw_outputs)}"
            )

        boxes = raw_outputs[0]
        classes = raw_outputs[1]
        scores = raw_outputs[2]
        num_arr = raw_outputs[3]

        if boxes.ndim != 3 or classes.ndim != 2 or scores.ndim != 2:
            raise InferenceError("Invalid output tensor shapes from detector")

        num_detections = int(num_arr[0])
        candidates: list[Detection] = []
        candidate_boxes_xywh: list[list[int]] = []
        candidate_scores: list[float] = []

        for i in range(min(num_detections, scores.shape[1])):
            conf = float(scores[0, i])
            if conf < self.config.confidence_threshold:
                continue

            class_id = int(classes[0, i])
            class_name = get_label(class_id)

            # Raw box is normalized [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = boxes[0, i]

            # Scale to original dimensions and clip
            x1 = int(np.clip(xmin * orig_width, 0, orig_width))
            y1 = int(np.clip(ymin * orig_height, 0, orig_height))
            x2 = int(np.clip(xmax * orig_width, 0, orig_width))
            y2 = int(np.clip(ymax * orig_height, 0, orig_height))

            # Discard degenerated boxes
            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1
            h = y2 - y1

            # Apparent-size: bounding box area as a percentage of full image area
            # Note: Strictly image area %. Not distance, proximity, or collision risk.
            orig_area = orig_width * orig_height
            box_area = w * h
            apparent_size = (
                round((box_area / orig_area) * 100.0, 2) if orig_area > 0 else 0.0
            )

            candidates.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    apparent_size_pct=apparent_size,
                )
            )
            candidate_boxes_xywh.append([x1, y1, w, h])
            candidate_scores.append(conf)

        if not candidates:
            return []

        # Apply secondary NMS if configured
        if self.config.nms_threshold > 0.0 and len(candidates) > 1:
            indices = cv2.dnn.NMSBoxes(
                bboxes=candidate_boxes_xywh,
                scores=candidate_scores,
                score_threshold=self.config.confidence_threshold,
                nms_threshold=self.config.nms_threshold,
            )
            if len(indices) > 0:
                filtered_indices = (
                    indices.flatten() if hasattr(indices, "flatten") else indices
                )
                return [candidates[idx] for idx in filtered_indices]

        return candidates

    def infer(self, frame: np.ndarray) -> tuple[list[Detection], float]:
        """
        Run inference on a single BGR camera frame.

        Returns:
            (detections, inference_latency_seconds)
        """
        if not self.is_loaded:
            raise InferenceError("Detector model is not loaded. Call load_model() first.")

        orig_h, orig_w = frame.shape[:2]
        tensor = self.preprocess(frame)

        t0 = time.perf_counter()
        try:
            raw_outputs = self._session.run(None, {self._input_name: tensor})  # type: ignore[union-attr]
        except Exception as e:
            raise InferenceError(f"Inference execution failed: {e}") from e
        latency = time.perf_counter() - t0

        detections = self.postprocess(raw_outputs, orig_w, orig_h)
        return detections, latency

    @staticmethod
    def count_by_class(detections: list[Detection]) -> dict[str, int]:
        """Aggregate detection counts grouped by class name."""
        counts: dict[str, int] = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    @staticmethod
    def annotate(
        frame: np.ndarray,
        detections: list[Detection],
        inference_fps: Optional[float] = None,
        capture_fps: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render bounding boxes, class labels, confidences, and telemetry text.
        """
        annotated = frame.copy()

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = (0, 255, 64)  # High-visibility green
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_text = f"{det.class_name} {det.confidence:.2f} ({det.apparent_size_pct:.1f}% area)"
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 2),
                (x1 + tw + 4, label_y + baseline - 2),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw telemetry HUD overlay in top-left
        hud_lines: list[str] = []
        if capture_fps is not None:
            hud_lines.append(f"Camera: {capture_fps:.1f} FPS")
        if latency_ms is not None:
            hud_lines.append(f"Inference: {latency_ms:.1f} ms")
        if inference_fps is not None:
            hud_lines.append(f"Infer rate: {inference_fps:.1f} FPS")

        counts = ObjectDetector.count_by_class(detections)
        if counts:
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            hud_lines.append(f"Objects: {summary}")
        else:
            hud_lines.append("Objects: 0")

        y_offset = 24
        for line in hud_lines:
            cv2.putText(
                annotated,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y_offset += 24

        return annotated

