# CV Dataset Pipeline

A streamlined Python pipeline for image dataset analysis, augmentation, stratified splitting, and tensor preparation.

## Features

- **Dataset Audit**: Validates image files, detects corruption, and reports class distributions.
- **Augmentations**: Horizontal flipping, rotation, and brightness adjustments.
- **Stratified Split**: Reproducible 70% / 15% / 15% (Train / Val / Test) split preserving class ratios.
- **Tensor Loader**: Resizes to `128x128`, normalizes to `[0.0, 1.0]`, and produces NumPy arrays.
- **Visual Analytics**: High-quality Matplotlib figures and terminal reporting.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete pipeline interactively:
```bash
python main.py
```

Run non-interactively and save figures to `plots/`:
```bash
python main.py --save-plots --skip-viz
```

Download dataset images from Open Images V7:
```bash
python download_dataset.py --count 40 --seed 42
```

## CLI Options

| Flag | Description | Default |
|---|---|---|
| `--dataset-dir` | Path to dataset directory | `dataset` |
| `--save-plots` | Export plots to `plots/` directory | `False` |
| `--skip-viz` | Skip interactive display windows | `False` |
