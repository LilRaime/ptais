# Lab 4: ONNX Export, OpenVINO Inference & Performance Benchmarking

Inference optimization pipeline for Computer Vision models (`MobileNetV3-Small` / `SimpleCNN`): ONNX export, graph validation, numerical parity testing, standalone ONNX Runtime and OpenVINO pipelines, and multi-engine benchmarking.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Run (All Steps)

Execute the full end-to-end pipeline (steps 1–7) in a single command:

```bash
python main.py
```

---

## Step-by-Step Usage

### 1. Export PyTorch to ONNX
Exports `../lab3/weights/mobilenet_v3_finetuned.pth` to ONNX with dynamic batch support:
```bash
python export_onnx.py
```

### 2. Verify ONNX Graph
Validates model integrity via `onnx.checker` and inspects inputs, outputs, and operators:
```bash
python verify_onnx.py
```

### 3. Compare PyTorch vs ONNX Runtime Outputs
Measures numerical divergence (MAE, RMSE, Max Diff, Cosine Similarity) on identical input:
```bash
python compare_outputs.py
```

### 4. Standalone ONNX Runtime Inference
Zero-PyTorch inference pipeline with pure NumPy/OpenCV preprocessing and HUD overlay:
```bash
python inference_onnx.py --source ../lab1/dataset/cat/cat_001.jpg --save plots/onnx_results --no-gui
```

### 5. Native OpenVINO Inference
Runs the exported ONNX model directly through OpenVINO Core:
```bash
python inference_openvino.py --source ../lab1/dataset/cat/cat_001.jpg --save plots/openvino_results --no-gui
```

### 6 & 7. Unified Benchmark & Report
Runs warm-up iterations, measures Latency, Throughput (FPS), and RAM usage, and generates comparison plots:
```bash
python benchmark.py --iterations 150 --warmup 25
```

---

## Benchmark Summary (CPU)

| Engine | Mean Latency | Throughput | Speedup | Memory RSS |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch (Eager)** | ~6.0–6.4 ms | ~160 FPS | **1.0x** (Baseline) | ~378 MB |
| **ONNX Runtime** | ~0.7–0.8 ms | ~1300 FPS | **~8.0x** | ~398 MB |
| **OpenVINO** | ~0.8–0.9 ms | ~1150 FPS | **~7.2x** | ~445 MB |

> Detailed metrics, plots, and architecture explanations are saved to `plots/benchmark_comparison.png` and `benchmark_report.md`.
