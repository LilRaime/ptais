# Programming Technologies for Artificial Intelligence Systems (PTAIS)

End-to-end Deep Learning and Computer Vision curriculum: from dataset curation and model training to production export, INT8 quantization, and hardware acceleration.

---

## Repository Structure

| Module | Topic | Key Technologies | Description |
| :--- | :--- | :--- | :--- |
| **[lab1/](lab1/)** | Dataset Collection & EDA | `Open Images V7`, `OpenCV`, `Pillow` | Dataset downloading, class balancing (`cat`, `dog`, `raven`), augmentations, and exploratory data analysis. |
| **[lab2/](lab2/)** | CNN Training from Scratch | `PyTorch`, `TorchVision` | Architecture design of `SimpleCNN`, training loop with early stopping, cross-entropy loss, and metrics tracking. |
| **[lab3/](lab3/)** | Transfer Learning & Fine-Tuning | `MobileNetV3-Small`, `ResNet18` | Backbone feature extraction, fine-tuning pretrained weights, learning rate scheduling, and comparative evaluation. |
| **[lab4/](lab4/)** | ONNX Export & Multi-Engine Inference | `ONNX`, `ONNX Runtime`, `OpenVINO` | Exporting PyTorch models to ONNX Opset 18, graph checker validation, numerical parity testing, and CPU benchmarking. |
| **[rgr/](rgr/)** | Optimization, Quantization & Pipeline | `INT8 Quantization`, `OpenVINO GPU` | INT8 dynamic & static quantization, end-to-end pipeline deployment, bottleneck analysis, and CPU/GPU/NPU device audit. |

---

## Installation & Setup

### Unified Environment Setup

This project uses `pyproject.toml` (PEP 517 / PEP 621) for dependency management:

```bash
# 1. Clone repository
git clone https://github.com/LilRaime/ptais.git
cd ptais

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install project and dependencies
pip install -e .
```

Alternatively, each laboratory directory includes an isolated `requirements.txt` if standalone execution is preferred.

---

## Quick Navigation

### Laboratory 1: Dataset Acquisition & Augmentation
```bash
cd lab1
python main.py
```

### Laboratory 2: Custom CNN Training
```bash
cd lab2
python main.py
```

### Laboratory 3: Transfer Learning & Benchmarking
```bash
cd lab3
python train_transfer.py
python compare.py
```

### Laboratory 4: ONNX Runtime & OpenVINO Benchmarking
```bash
cd lab4
python main.py
```

### Calculation & Graphic Project (RGR): Quantization & Pipeline
```bash
cd rgr
python main.py
```

---

## Technical Stack
- **Deep Learning Frameworks:** `PyTorch 2.x`, `TorchVision`
- **Interoperability & Formats:** `ONNX Opset 18`, `onnxscript`
- **Inference Accelerators:** `ONNX Runtime`, `Intel OpenVINO`
- **Image Processing & Vision:** `OpenCV`, `Pillow`, `NumPy`
- **Profiling & System Metrics:** `psutil`, `matplotlib`, `rich`

---

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
