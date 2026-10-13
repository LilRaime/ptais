# Lab 2: Cat / Dog / Raven Classification (PyTorch CNN)

Image classification of 3 animal classes using a Convolutional Neural Network.

## CNN Architecture
- **Conv 1**: `Conv2d(3, 32, 3)` → `BatchNorm` → `ReLU` → `MaxPool2d(2, 2)` (128×128 → 64×64)
- **Conv 2**: `Conv2d(32, 64, 3)` → `BatchNorm` → `ReLU` → `MaxPool2d(2, 2)` (64×64 → 32×32)
- **Pool**: `AdaptiveAvgPool2d((2, 2))` (32×32 → 2×2)
- **Classifier**: `Flatten` → `Linear(256, 128)` → `ReLU` → `Dropout(0.3)` → `Linear(128, 3)` (~52K parameters, fast CPU inference)

## Setup & Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## CLI Arguments
- `--dataset-dir`: path to dataset folder (default: `../lab1/dataset`)
- `--epochs`: number of training epochs (default: `25`)
- `--batch-size`: batch size for DataLoader (default: `16`)
- `--lr`: learning rate (default: `0.001`)
- `--device`: execution device (default: `cpu`)
- `--seed`: random seed (default: `42`)
- `--no-preload`: disable RAM caching of dataset images (enabled by default for high CPU speed)

## Outputs
- `best_model.pth`: checkpoint with best validation accuracy
- `plots/loss_accuracy_curves.png`: train & validation loss / accuracy curves
- `plots/confusion_matrix.png`: test confusion matrix heatmap
- `plots/predictions_analysis.png`: 5 correct and 5 incorrect predictions with explanations
