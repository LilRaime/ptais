# Calculation and Graphic Project (RGR): Computer Vision Model Optimization & Quantization

Development, quantization, and evaluation pipeline for `MobileNetV3-Small`:
1. Base **FP32** model export and ONNX graph verification (from Lab 4).
2. **INT8** Quantization (Dynamic and dataset-calibrated Static Quantization).
3. Comprehensive model comparison: Accuracy, model compression, Latency, FPS, and RAM usage.
4. Integrated end-to-end pipeline: Data source -> Preprocessing -> Inference -> Postprocessing -> Output.
5. Stage-by-stage latency profiling and Bottleneck Analysis.
6. Hardware audit and device benchmarking across CPU / GPU / NPU.
7. Justified configuration selection matrix for industrial deployment scenarios.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Run (Full Pipeline)

Execute all pipeline stages in a single command:

```bash
python main.py
```

---

## Step-by-Step Execution

### 1. Export Base FP32 ONNX Model
```bash
python export_fp32.py
```

### 2. Quantize Model to INT8 (Dynamic & Static)
```bash
python quantize_model.py
```

### 3. Compare FP32 vs INT8 (Accuracy, Speed, Memory)
```bash
python compare_models.py --iterations 150 --warmup 25
```

### 4 & 5. Integrated Pipeline & Bottleneck Profiling
```bash
python run_pipeline.py --source ../lab1/dataset --save plots/pipeline_results
```

### 6. Hardware Audit & Benchmark (CPU / GPU / NPU)
```bash
python benchmark_devices.py
```

### 7. Generate Final Engineering Report
```bash
python generate_report.py
```

---

## Output & Artifacts
- Final engineering report: `benchmark_report.md`.
- Benchmark charts and visual analytics: `plots/` directory.
- Annotated pipeline inference frames: `plots/pipeline_results/`.
