import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from src.benchmark import benchmark_inference
from src.dataset import create_dataloaders
from src.models import get_model, count_parameters
from src.pipeline import ImageStream, preprocess_frame

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    default_ds = Path("../lab1/dataset") if Path("../lab1/dataset").exists() else Path("lab1/dataset")
    parser.add_argument("--dataset-dir", type=Path, default=default_ds)
    parser.add_argument("--cfg1-name", type=str, default="simple_cnn")
    parser.add_argument("--cfg1-weights", type=Path, default=Path("../lab2/best_model.pth"))
    parser.add_argument("--cfg1-size", type=int, default=128)
    parser.add_argument("--cfg2-name", type=str, default="mobilenet_v3_small")
    parser.add_argument("--cfg2-weights", type=Path, default=Path("weights/mobilenet_v3_finetuned.pth"))
    parser.add_argument("--cfg2-size", type=int, default=224)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    parser.add_argument("--show-plots", action="store_true")
    return parser.parse_args()

def load_configured_model(name, weights_path, num_classes, device):
    model = get_model(name=name, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        console.print(f"Loaded weights for [bold]{name}[/bold] from {weights_path}")
    else:
        console.print(f"[yellow]Warning: Weights {weights_path} not found. Running with initial weights.[/yellow]")
    model.to(device).eval()
    return model

def evaluate_accuracy(model, dataset_dir, img_size, batch_size=16, device="cpu"):
    _, _, test_loader, _, _ = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        img_size=(img_size, img_size),
        preload=True
    )
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

def plot_comparisons(cfg1_info, cfg2_info, plots_dir, show_plots=False):
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    labels = [cfg1_info["label"], cfg2_info["label"]]
    colors = ["#3498db", "#2ecc71"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    latencies = [cfg1_info["mean_ms"], cfg2_info["mean_ms"]]
    yerr = [cfg1_info["std_ms"], cfg2_info["std_ms"]]
    bars1 = axes[0, 0].bar(labels, latencies, yerr=yerr, capsize=6, color=colors, alpha=0.88, edgecolor="black")
    axes[0, 0].set_title("Average Inference Latency (ms, lower is better)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Latency (ms)")
    for b in bars1:
        axes[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.2f} ms", ha="center", va="bottom", fontweight="bold")

    fps_vals = [cfg1_info["fps"], cfg2_info["fps"]]
    bars2 = axes[0, 1].bar(labels, fps_vals, color=colors, alpha=0.88, edgecolor="black")
    axes[0, 1].set_title("Inference FPS (higher is better)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("FPS")
    for b in bars2:
        axes[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", va="bottom", fontweight="bold")

    params_vals = [cfg1_info["params"] / 1000.0, cfg2_info["params"] / 1000.0]
    bars3 = axes[1, 0].bar(labels, params_vals, color=colors, alpha=0.88, edgecolor="black")
    axes[1, 0].set_title("Trainable Parameters (x1000)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Parameters (k)")
    for b in bars3:
        axes[1, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{b.get_height():.1f}k", ha="center", va="bottom", fontweight="bold")

    acc_vals = [cfg1_info["accuracy"] * 100.0, cfg2_info["accuracy"] * 100.0]
    bars4 = axes[1, 1].bar(labels, acc_vals, color=colors, alpha=0.88, edgecolor="black")
    axes[1, 1].set_title("Test Set Accuracy (%)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_ylim(0, 105)
    for b in bars4:
        axes[1, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    metrics_path = plots_dir / "comparison_metrics.png"
    plt.savefig(metrics_path, dpi=300)
    plt.close()

    fig2, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=[cfg1_info["raw_latencies"], cfg2_info["raw_latencies"]], ax=ax, palette=colors, width=0.45)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_title(f"Latency Distribution over {cfg1_info['iterations']} Runs (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms)")
    dist_path = plots_dir / "latency_distribution.png"
    plt.tight_layout()
    plt.savefig(dist_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close()

    return metrics_path, dist_path

def main():
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    root = Path(args.dataset_dir)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    num_classes = len(classes)

    model1 = load_configured_model(args.cfg1_name, args.cfg1_weights, num_classes, device)
    model2 = load_configured_model(args.cfg2_name, args.cfg2_weights, num_classes, device)

    image_stream = ImageStream(args.dataset_dir)
    console.print(f"Loaded [bold green]{len(image_stream)}[/bold green] images for benchmarking.")
    tensors1 = [preprocess_frame(frame, img_size=(args.cfg1_size, args.cfg1_size), device=device) for frame, _ in image_stream]
    tensors2 = [preprocess_frame(frame, img_size=(args.cfg2_size, args.cfg2_size), device=device) for frame, _ in image_stream]

    console.print(f"Benchmarking [bold]{args.cfg1_name}[/bold] ({args.cfg1_size}x{args.cfg1_size}) over {args.iterations} runs...")
    res1 = benchmark_inference(model1, tensors1, iterations=args.iterations, warmup_iterations=args.warmup)

    console.print(f"Benchmarking [bold]{args.cfg2_name}[/bold] ({args.cfg2_size}x{args.cfg2_size}) over {args.iterations} runs...")
    res2 = benchmark_inference(model2, tensors2, iterations=args.iterations, warmup_iterations=args.warmup)

    console.print("Evaluating test set accuracy for both configurations...")
    acc1 = evaluate_accuracy(model1, args.dataset_dir, args.cfg1_size, device=args.device)
    acc2 = evaluate_accuracy(model2, args.dataset_dir, args.cfg2_size, device=args.device)

    cfg1_info = {
        "label": f"{args.cfg1_name}\n({args.cfg1_size}x{args.cfg1_size})",
        "mean_ms": res1["mean_ms"],
        "std_ms": res1["std_ms"],
        "median_ms": res1["median_ms"],
        "p90_ms": res1["p90_ms"],
        "p99_ms": res1["p99_ms"],
        "fps": res1["fps"],
        "params": count_parameters(model1),
        "accuracy": acc1,
        "raw_latencies": res1["latencies"],
        "iterations": args.iterations
    }

    cfg2_info = {
        "label": f"{args.cfg2_name}\n({args.cfg2_size}x{args.cfg2_size})",
        "mean_ms": res2["mean_ms"],
        "std_ms": res2["std_ms"],
        "median_ms": res2["median_ms"],
        "p90_ms": res2["p90_ms"],
        "p99_ms": res2["p99_ms"],
        "fps": res2["fps"],
        "params": count_parameters(model2),
        "accuracy": acc2,
        "raw_latencies": res2["latencies"],
        "iterations": args.iterations
    }

    table = Table(title="Configuration Comparison & Performance Benchmark", border_style="bright_blue")
    table.add_column("Metric", style="bold cyan")
    table.add_column(f"Config 1: {args.cfg1_name} ({args.cfg1_size}x{args.cfg1_size})", justify="right")
    table.add_column(f"Config 2: {args.cfg2_name} ({args.cfg2_size}x{args.cfg2_size})", justify="right")

    table.add_row("Parameters", f"{cfg1_info['params']:,}", f"{cfg2_info['params']:,}")
    table.add_row("Accuracy (Test Set)", f"{cfg1_info['accuracy'] * 100:.2f}%", f"{cfg2_info['accuracy'] * 100:.2f}%")
    table.add_row("Warmup Iterations", str(args.warmup), str(args.warmup))
    table.add_row("Benchmark Iterations", str(args.iterations), str(args.iterations))
    table.add_row("Mean Latency", f"{cfg1_info['mean_ms']:.2f} ms", f"{cfg2_info['mean_ms']:.2f} ms")
    table.add_row("Std Latency", f"±{cfg1_info['std_ms']:.2f} ms", f"±{cfg2_info['std_ms']:.2f} ms")
    table.add_row("Median (p50) Latency", f"{cfg1_info['median_ms']:.2f} ms", f"{cfg2_info['median_ms']:.2f} ms")
    table.add_row("90th Percentile Latency", f"{cfg1_info['p90_ms']:.2f} ms", f"{cfg2_info['p90_ms']:.2f} ms")
    table.add_row("99th Percentile Latency", f"{cfg1_info['p99_ms']:.2f} ms", f"{cfg2_info['p99_ms']:.2f} ms")
    table.add_row("Inference FPS", f"[bold]{cfg1_info['fps']:.1f}[/bold]", f"[bold]{cfg2_info['fps']:.1f}[/bold]")

    console.print(table)

    p1, p2 = plot_comparisons(cfg1_info, cfg2_info, args.plots_dir, args.show_plots)
    console.print(f"Saved comparison plots:\n- {p1}\n- {p2}")

if __name__ == "__main__":
    main()
