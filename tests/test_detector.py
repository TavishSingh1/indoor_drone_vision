"""
test_detector.py — Comprehensive unit and integration tests for ObjectDetector.

Tests:
1. Preprocessing format, shape, and type
2. Synthetic output decoding, bounding box coordinate scaling, and class mapping
3. Confidence threshold filtering and secondary NMS
4. Object count aggregation by class
5. Model loading validation and error handling for missing files
6. End-to-end inference on real saved test image (if model file exists)
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

from src.detector import (
    Detection,
    DetectorConfig,
    InferenceError,
    ModelLoadError,
    ObjectDetector,
)
from src.labels import COCO_LABELS_91, get_label


class TestDetectorLabels(unittest.TestCase):
    """Test label mapping."""

    def test_known_labels(self):
        self.assertEqual(get_label(1), "person")
        self.assertEqual(get_label(44), "bottle")
        self.assertEqual(get_label(62), "chair")
        self.assertEqual(get_label(67), "dining table")
        self.assertEqual(get_label(73), "laptop")
        self.assertEqual(get_label(27), "backpack")

    def test_unknown_label(self):
        self.assertEqual(get_label(999), "class_999")


class TestDetectorPreprocessing(unittest.TestCase):
    """Test preprocessing logic."""

    def setUp(self):
        self.config = DetectorConfig(input_width=300, input_height=300)
        self.detector = ObjectDetector(self.config)

    def test_preprocess_shape_and_dtype(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Put arbitrary values in channels
        dummy_frame[:, :, 0] = 50   # Blue
        dummy_frame[:, :, 1] = 100  # Green
        dummy_frame[:, :, 2] = 150  # Red

        tensor = self.detector.preprocess(dummy_frame)
        self.assertEqual(tensor.shape, (1, 300, 300, 3))
        self.assertEqual(tensor.dtype, np.uint8)

        # Preprocessing converts BGR to RGB
        self.assertEqual(tensor[0, 0, 0, 0], 150)  # Red
        self.assertEqual(tensor[0, 0, 0, 1], 100)  # Green
        self.assertEqual(tensor[0, 0, 0, 2], 50)   # Blue

    def test_preprocess_empty_frame_raises(self):
        with self.assertRaises(ValueError):
            self.detector.preprocess(np.empty((0, 0, 3), dtype=np.uint8))


class TestDetectorPostprocessing(unittest.TestCase):
    """Test decoding of raw model outputs with synthetic tensors."""

    def setUp(self):
        self.config = DetectorConfig(
            confidence_threshold=0.5,
            nms_threshold=0.5,
        )
        self.detector = ObjectDetector(self.config)

    def test_postprocess_decoding(self):
        orig_w = 640
        orig_h = 480

        # Normalized: [ymin, xmin, ymax, xmax]
        # Box 1: (ymin=0.1, xmin=0.2, ymax=0.5, xmax=0.6) -> class 1 (person), score 0.90
        # Box 2: (ymin=0.6, xmin=0.1, ymax=0.9, xmax=0.4) -> class 62 (chair), score 0.75
        # Box 3: low confidence score 0.35 (should be filtered out)
        synthetic_boxes = np.zeros((1, 10, 4), dtype=np.float32)
        synthetic_boxes[0, 0] = [0.1, 0.2, 0.5, 0.6]
        synthetic_boxes[0, 1] = [0.6, 0.1, 0.9, 0.4]
        synthetic_boxes[0, 2] = [0.0, 0.0, 1.0, 1.0]

        synthetic_classes = np.zeros((1, 10), dtype=np.float32)
        synthetic_classes[0, 0] = 1.0   # person
        synthetic_classes[0, 1] = 62.0  # chair
        synthetic_classes[0, 2] = 44.0  # bottle

        synthetic_scores = np.zeros((1, 10), dtype=np.float32)
        synthetic_scores[0, 0] = 0.90
        synthetic_scores[0, 1] = 0.75
        synthetic_scores[0, 2] = 0.35  # below threshold

        synthetic_num = np.array([3.0], dtype=np.float32)

        raw_outputs = [
            synthetic_boxes,
            synthetic_classes,
            synthetic_scores,
            synthetic_num,
        ]

        detections = self.detector.postprocess(raw_outputs, orig_w, orig_h)
        self.assertEqual(len(detections), 2)

        det1 = detections[0]
        self.assertEqual(det1.class_name, "person")
        self.assertEqual(det1.class_id, 1)
        self.assertAlmostEqual(det1.confidence, 0.90, places=2)
        # Expected: xmin = 0.2*640 = 128, ymin = 0.1*480 = 48
        #           xmax = 0.6*640 = 384, ymax = 0.5*480 = 240
        self.assertEqual(det1.box, (128, 48, 384, 240))
        # Apparent size: (256 * 192) / (640 * 480) * 100 = 16.0%
        self.assertAlmostEqual(det1.apparent_size_pct, 16.0, places=1)

        det2 = detections[1]
        self.assertEqual(det2.class_name, "chair")
        self.assertEqual(det2.class_id, 62)
        self.assertAlmostEqual(det2.confidence, 0.75, places=2)
        self.assertEqual(det2.box, (64, 288, 256, 432))
        # Apparent size: (192 * 144) / (640 * 480) * 100 = 9.0%
        self.assertAlmostEqual(det2.apparent_size_pct, 9.0, places=1)

    def test_count_by_class(self):
        dets = [
            Detection(box=(0, 0, 10, 10), class_id=1, class_name="person", confidence=0.9),
            Detection(box=(0, 0, 10, 10), class_id=1, class_name="person", confidence=0.8),
            Detection(box=(0, 0, 10, 10), class_id=62, class_name="chair", confidence=0.7),
        ]
        counts = ObjectDetector.count_by_class(dets)
        self.assertEqual(counts, {"person": 2, "chair": 1})

    def test_missing_outputs_raises(self):
        with self.assertRaises(InferenceError):
            self.detector.postprocess([], 640, 480)


class TestDetectorModelIntegration(unittest.TestCase):
    """Test actual model file loading and inference if present."""

    def test_missing_model_raises(self):
        cfg = DetectorConfig(model_path="nonexistent_model.onnx")
        det = ObjectDetector(cfg)
        with self.assertRaises(ModelLoadError):
            det.load_model()

    def test_load_and_infer_real_model(self):
        model_path = Path("models/ssd_mobilenet_v1.onnx")
        if not model_path.exists():
            self.skipTest("Model file not downloaded yet")

        cfg = DetectorConfig(
            model_path=str(model_path),
            confidence_threshold=0.3,
            num_threads=2,
        )
        detector = ObjectDetector(cfg)
        detector.load_model()
        self.assertTrue(detector.is_loaded)

        test_img_path = Path("tests/test_person.jpg")
        if test_img_path.exists():
            import cv2
            img = cv2.imread(str(test_img_path))
            detections, latency = detector.infer(img)
            self.assertGreater(latency, 0.0)
            self.assertIsInstance(detections, list)

            # Person should be detected in test_person.jpg
            class_names = [d.class_name for d in detections]
            self.assertIn("person", class_names)

            annotated = ObjectDetector.annotate(
                img, detections, latency_ms=latency * 1000
            )
            self.assertEqual(annotated.shape, img.shape)


if __name__ == "__main__":
    unittest.main()

