# Lab 3: Pretrained Models, Transfer Learning, and OpenCV Image Pipeline

Computer Vision pipeline integrating pretrained architectures (`MobileNetV3-Small`, `ResNet18`, `SimpleCNN`), Transfer Learning, OpenCV image pipeline, and latency/FPS benchmarking.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Transfer Learning Training

Fine-tunes a pretrained MobileNetV3-Small on the dataset (`cat`, `dog`, `raven`):

```bash
python train_transfer.py --model-name mobilenet_v3_small --epochs 10 --batch-size 16 --img-size 224
```

Outputs:
- Saved checkpoint: `weights/mobilenet_v3_finetuned.pth`

## 2. OpenCV Image Inference Pipeline

Reads individual image files or entire image directories via OpenCV, performs preprocessing/postprocessing, and renders HUD overlays (predicted class, confidence, probability distribution, latency, FPS):

- **Single image (without saving):**
  ```bash
  python inference.py --source ../lab1/dataset/cat/cat_001.jpg
  ```

- **Directory of images:**
  ```bash
  python inference.py --source ../lab1/dataset/cat
  ```

- **Save annotated images (to plots/inference_results by default):**
  ```bash
  python inference.py --source ../lab1/dataset/dog --save
  ```

- **Headless mode with saving:**
  ```bash
  python inference.py --source ../lab1/dataset/raven --save --no-gui
  ```

## 3. Benchmark & Configuration Comparison

Compares two configurations (e.g. `SimpleCNN` from Lab 2 vs `MobileNetV3-Small` Transfer Learning):
- Loads real images through OpenCV pipeline.
- Performs warmup passes.
- Measures inference latency and FPS over $\ge 100$ iterations.
- Evaluates test accuracy.
- Generates plots and metric summary tables.

```bash
python compare.py --cfg1-name simple_cnn --cfg1-weights ../lab2/best_model.pth --cfg1-size 128 --cfg2-name mobilenet_v3_small --cfg2-weights weights/mobilenet_v3_finetuned.pth --cfg2-size 224 --iterations 100
```

Outputs:
- `plots/comparison_metrics.png`: bar charts comparing Latency, FPS, Parameters, and Accuracy
- `plots/latency_distribution.png`: box plot showing latency distribution across iterations
