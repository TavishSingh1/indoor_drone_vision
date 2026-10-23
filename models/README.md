# Pretrained Object Detection Models

This directory contains verified, publicly available, lightweight object detection models trained on MS COCO and exported in ONNX format.

> **Safety:** Observational vision only. Independent of flight controllers, telemetry ports, motors, and RC commands.

---

## 1. Verified Model Assets

### Model A: MobileNet-SSD v1 FP32 (Baseline Fallback)
- **Filename**: `ssd_mobilenet_v1.onnx`
- **Source**: [Hugging Face onnxmodelzoo/ssd_mobilenet_v1_12](https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12)
- **Direct Asset URL**: `https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12/resolve/main/ssd_mobilenet_v1_12.onnx`
- **License**: **MIT License**
- **Opset**: 12
- **File Size**: 29,461,902 bytes (~29.4 MB)
- **SHA-256 Checksum**:
  `b8fba5e404077d4048d27fcd1667e85e27e192eb9bf51e696c46a3acd7d21058`

### Model B: MobileNet-SSD v1 INT8 (Quantized Performance Asset)
- **Filename**: `ssd_mobilenet_v1_int8.onnx`
- **Source**: [Hugging Face onnxmodelzoo/ssd_mobilenet_v1_12-int8](https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12-int8)
- **Direct Asset URL**: `https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12-int8/resolve/main/ssd_mobilenet_v1_12-int8.onnx`
- **License**: **MIT License**
- **Opset**: 12
- **File Size**: 9,540,809 bytes (~9.5 MB, 68% size reduction)
- **SHA-256 Checksum**:
  `2b79e6a7fb1ec6a33f332b9b10d82d9de4b7b49dcd26b5946921bb356895c954`

---

## 2. Tensor Specifications (Shared Across Both Models)

### Inputs
- **Tensor Name**: `inputs`
- **Data Type**: `tensor(uint8)`
- **Shape**: `[1, 300, 300, 3]` (NHWC layout)
- **Color Format**: RGB uint8 (`0..255`)

### Outputs
1. **`detection_boxes`**: `float32 [1, 100, 4]` — Normalized `[ymin, xmin, ymax, xmax]` coordinates (`0.0 .. 1.0`).
2. **`detection_classes`**: `float32 [1, 100]` — 1-based COCO class identifiers (`1 .. 90`).
3. **`detection_scores`**: `float32 [1, 100]` — Detection confidence probabilities (`0.0 .. 1.0`).
4. **`num_detections`**: `float32 [1]` — Total candidate count from internal NMS.

---

## 3. Measured Performance Comparison (Raspberry Pi 4 CPU, 2 Threads)

Tested under identical conditions on `tests/test_person.jpg` (640×480 downscaled to 300×300):

| Metric | FP32 (`ssd_mobilenet_v1.onnx`) | INT8 (`ssd_mobilenet_v1_int8.onnx`) | Improvement |
|---|---|---|---|
| **Model Size** | 29.4 MB | **9.1 MB** | **68.9% smaller** |
| **Model Load Time** | 51.51 s | 53.37 s | Similar startup time |
| **Mean Inference Latency** | 423.5 ms | **254.2 ms** | **40.0% latency reduction** |
| **Inference Rate** | 2.36 FPS | **3.93 FPS** | **+66.5% higher throughput** |
| **Person Detection Conf** | 0.925 | **0.955** | Slightly sharper |
| **Ball Detection Conf** | 0.372 | **0.383** | Equivalent |
| **Detected Candidates** | 3 objects | 3 objects | 100% matched accuracy |

---

## 4. Supported Indoor Target Classes (COCO 91 taxonomy)
- `person` (ID 1)
- `backpack` (ID 27)
- `bottle` (ID 44)
- `chair` (ID 62)
- `couch` (ID 63)
- `dining table` (ID 67)
- `tv` (ID 72)
- `laptop` (ID 73)
- `cell phone` (ID 77)
- `book` (ID 84)
