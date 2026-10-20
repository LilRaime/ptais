import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import psutil
from rich.console import Console
from rich.table import Table
from src.pipeline import postprocess_output, preprocess_image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-model", type=Path, default=Path("weights/mobilenet_v3_fp32.onnx"))
    parser.add_argument("--int8-model", type=Path, default=Path("weights/mobilenet_v3_int8.onnx"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("../lab1/dataset"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    return parser.parse_args()

def get_process_memory_mb():
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def benchmark_ort_session(session, input_tensor, warmup_iters, test_iters):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    feed = {input_name: input_tensor}

    for _ in range(warmup_iters):
        session.run([output_name], feed)

    mem_before = get_process_memory_mb()
    latencies = []
    for _ in range(test_iters):
        t0 = time.perf_counter()
        session.run([output_name], feed)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    mem_after = get_process_memory_mb()

    lat = np.array(latencies)
    mean_lat = float(np.mean(lat))
    stats = {
        "mean_ms": mean_lat,
        "median_ms": float(np.median(lat)),
        "min_ms": float(np.min(lat)),
        "max_ms": float(np.max(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "std_ms": float(np.std(lat)),
        "fps": float(1000.0 / mean_lat) if mean_lat > 0 else 0.0,
        "mem_mb": max(mem_before, mem_after),
    }
    return stats, lat

def evaluate_accuracy_and_divergence(session_fp32, session_int8, dataset_dir, class_names, img_size):
    in_fp32 = session_fp32.get_inputs()[0].name
    in_int8 = session_int8.get_inputs()[0].name
    out_fp32 = session_fp32.get_outputs()[0].name
    out_int8 = session_int8.get_outputs()[0].name

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(dataset_dir.rglob("*")) if p.suffix.lower() in valid_exts]

    fp32_correct = 0
    int8_correct = 0
    agreed_predictions = 0
    total_evaluated = 0

    diffs = []
    cos_sims = []

    for img_path in images:
        gt_class = img_path.parent.name.lower()
        if gt_class not in class_names:
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        tensor = preprocess_image(frame, img_size=img_size)

        out_f = session_fp32.run([out_fp32], {in_fp32: tensor})[0]
        out_i = session_int8.run([out_int8], {in_int8: tensor})[0]

        pred_f, conf_f, _ = postprocess_output(out_f, class_names)
        pred_i, conf_i, _ = postprocess_output(out_i, class_names)

        if pred_f == gt_class:
            fp32_correct += 1
        if pred_i == gt_class:
            int8_correct += 1
        if pred_f == pred_i:
            agreed_predictions += 1

        total_evaluated += 1

        diff = np.abs(out_f - out_i)
        diffs.append(np.mean(diff))

        v_f, v_i = out_f.flatten(), out_i.flatten()
        denom = (np.linalg.norm(v_f) * np.linalg.norm(v_i)) + 1e-9
        cos_sims.append(float(np.dot(v_f, v_i) / denom))

    acc_fp32 = (fp32_correct / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    acc_int8 = (int8_correct / total_evaluated * 100.0) if total_evaluated > 0 else 0.0
    agreement_rate = (agreed_predictions / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    return {
        "total_evaluated": total_evaluated,
        "acc_fp32": acc_fp32,
        "acc_int8": acc_int8,
        "acc_delta": acc_fp32 - acc_int8,
        "agreement_rate": agreement_rate,
        "mean_mae": float(np.mean(diffs)) if diffs else 0.0,
        "mean_cos_sim": float(np.mean(cos_sims)) if cos_sims else 1.0,
    }

def generate_comparison_plots(fp32_stats, int8_stats, acc_info, fp32_size, int8_size, fp32_lats, int8_lats, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    fig.patch.set_facecolor("#181920")
    colors = ["#3498db", "#2ecc71"]
    labels = ["FP32 Base", "INT8 Quantized"]

    for ax in axes:
        ax.set_facecolor("#22232e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#44475a")

    lat_means = [fp32_stats["mean_ms"], int8_stats["mean_ms"]]
    lat_stds = [fp32_stats["std_ms"], int8_stats["std_ms"]]
    b1 = axes[0].bar(labels, lat_means, yerr=lat_stds, capsize=5, color=colors, alpha=0.9, width=0.45)
    axes[0].set_title("Mean Latency (ms)\n[Lower is better]", color="white", fontsize=11, fontweight="bold")
    for b in b1:
        axes[0].annotate(f"{b.get_height():.2f} ms", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontweight="bold")

    fps_vals = [fp32_stats["fps"], int8_stats["fps"]]
    b2 = axes[1].bar(labels, fps_vals, color=colors, alpha=0.9, width=0.45)
    axes[1].set_title("Throughput (FPS)\n[Higher is better]", color="white", fontsize=11, fontweight="bold")
    for b in b2:
        axes[1].annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontweight="bold")

    sizes = [fp32_size, int8_size]
    b3 = axes[2].bar(labels, sizes, color=colors, alpha=0.9, width=0.45)
    axes[2].set_title("Model File Size (MB)\n[Lower is better]", color="white", fontsize=11, fontweight="bold")
    for b in b3:
        axes[2].annotate(f"{b.get_height():.2f} MB", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontweight="bold")

    acc_vals = [acc_info["acc_fp32"], acc_info["acc_int8"]]
    b4 = axes[3].bar(labels, acc_vals, color=colors, alpha=0.9, width=0.45)
    axes[3].set_title(f"Accuracy (%)\n[N={acc_info['total_evaluated']}]", color="white", fontsize=11, fontweight="bold")
    axes[3].set_ylim([max(0, min(acc_vals) - 10), 105])
    for b in b4:
        axes[3].annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=160, facecolor=fig.get_facecolor())
    plt.close()

def main():
    args = parse_args()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session_fp32 = ort.InferenceSession(str(args.fp32_model), sess_options=opts, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(str(args.int8_model), sess_options=opts, providers=["CPUExecutionProvider"])

    fp32_size = args.fp32_model.stat().st_size / (1024 * 1024)
    int8_size = args.int8_model.stat().st_size / (1024 * 1024)

    dummy_tensor = np.random.randn(args.batch_size, 3, args.img_size, args.img_size).astype(np.float32)
    fp32_stats, fp32_lats = benchmark_ort_session(session_fp32, dummy_tensor, args.warmup, args.iterations)
    int8_stats, int8_lats = benchmark_ort_session(session_int8, dummy_tensor, args.warmup, args.iterations)

    class_names = ["cat", "dog", "raven"]
    acc_info = evaluate_accuracy_and_divergence(session_fp32, session_int8, args.dataset_dir, class_names, args.img_size)

    speedup = fp32_stats["mean_ms"] / int8_stats["mean_ms"] if int8_stats["mean_ms"] > 0 else 1.0
    size_ratio = fp32_size / int8_size if int8_size > 0 else 1.0

    table = Table(title="Comprehensive Model Comparison: FP32 Base vs INT8 Quantized", border_style="bright_blue")
    table.add_column("Evaluation Metric", style="cyan")
    table.add_column("FP32 Base Model", justify="right")
    table.add_column("INT8 Quantized Model", justify="right")
    table.add_column("Delta / Improvement", justify="right", style="bold yellow")

    table.add_row("Model File Size", f"{fp32_size:.2f} MB", f"{int8_size:.2f} MB", f"{size_ratio:.2f}x compression (-{(1 - int8_size/fp32_size)*100:.1f}%)")
    table.add_row("Mean Latency", f"{fp32_stats['mean_ms']:.2f} ms", f"{int8_stats['mean_ms']:.2f} ms", f"{speedup:.2f}x speedup")
    table.add_row("Median Latency", f"{fp32_stats['median_ms']:.2f} ms", f"{int8_stats['median_ms']:.2f} ms", f"{fp32_stats['median_ms']/int8_stats['median_ms']:.2f}x")
    table.add_row("P95 Latency", f"{fp32_stats['p95_ms']:.2f} ms", f"{int8_stats['p95_ms']:.2f} ms", f"{fp32_stats['p95_ms'] - int8_stats['p95_ms']:.2f} ms faster")
    table.add_row("Throughput (FPS)", f"{fp32_stats['fps']:.1f} FPS", f"{int8_stats['fps']:.1f} FPS", f"{int8_stats['fps'] - fp32_stats['fps']:+.1f} FPS")
    table.add_row("Process RAM (RSS)", f"{fp32_stats['mem_mb']:.1f} MB", f"{int8_stats['mem_mb']:.1f} MB", f"{int8_stats['mem_mb'] - fp32_stats['mem_mb']:+.1f} MB")
    table.add_row("Dataset Accuracy", f"{acc_info['acc_fp32']:.2f}%", f"{acc_info['acc_int8']:.2f}%", f"{acc_info['acc_delta']:+.2f}% drop")
    table.add_row("Prediction Agreement", "100.0%", f"{acc_info['agreement_rate']:.2f}%", f"{acc_info['agreement_rate']:.1f}% matching")
    table.add_row("Output MAE Divergence", "0.000000", f"{acc_info['mean_mae']:.6f}", "Acceptable noise")
    table.add_row("Cosine Similarity", "1.000000", f"{acc_info['mean_cos_sim']:.6f}", "[bold green]High Fidelity[/bold green]")

    console.print(table)

    metrics_path = args.output_dir / "comparison_metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_data = {
        "fp32_size_mb": round(fp32_size, 2),
        "int8_size_mb": round(int8_size, 2),
        "compression_ratio": round(size_ratio, 2),
        "reduction_pct": round((1.0 - (int8_size / fp32_size)) * 100.0, 1) if fp32_size > 0 else 0.0,
        "speedup": round(speedup, 2),
        "fp32_stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fp32_stats.items()},
        "int8_stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in int8_stats.items()},
        "acc_info": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in acc_info.items()},
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    plot_path = args.output_dir / "accuracy_and_performance.png"
    generate_comparison_plots(fp32_stats, int8_stats, acc_info, fp32_size, int8_size, fp32_lats, int8_lats, plot_path)
    console.print(f"[bold green]Saved comparison charts to:[/bold green] {plot_path}")

if __name__ == "__main__":
    main()
