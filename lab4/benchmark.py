import argparse
import gc
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import openvino as ov
import psutil
import sys
import torch
from rich.console import Console
from rich.table import Table
from src.models import load_trained_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("../lab3/weights/mobilenet_v3_finetuned.pth"))
    parser.add_argument("--onnx", type=Path, default=Path("weights/mobilenet_v3_small.onnx"))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--report", type=Path, default=Path("benchmark_report.md"))
    return parser.parse_args()

def get_process_memory_mb():
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def measure_fn(fn, warmup_iters, test_iters):
    for _ in range(warmup_iters):
        fn()
    mem = get_process_memory_mb()
    latencies = []
    for _ in range(test_iters):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return np.array(latencies), mem

def compute_stats(lat, batch_size):
    mean_lat = float(np.mean(lat))
    return {
        "mean_ms": mean_lat,
        "median_ms": float(np.median(lat)),
        "std_ms": float(np.std(lat)),
        "min_ms": float(np.min(lat)),
        "max_ms": float(np.max(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "fps": float((batch_size * 1000.0) / mean_lat) if mean_lat > 0 else 0.0,
    }

def save_plots(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    engines = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#1e1e28")
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    means = [results[e]["stats"]["mean_ms"] for e in engines]
    stds = [results[e]["stats"]["std_ms"] for e in engines]
    axes[0].set_facecolor("#282a36")
    b1 = axes[0].bar(engines, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.5)
    axes[0].set_title("Mean Latency (ms) [Lower is better]", color="white")
    axes[0].tick_params(colors="white")
    for b in b1:
        axes[0].annotate(f"{b.get_height():.2f} ms", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontweight="bold")

    fps_vals = [results[e]["stats"]["fps"] for e in engines]
    axes[1].set_facecolor("#282a36")
    b2 = axes[1].bar(engines, fps_vals, color=colors, alpha=0.85, width=0.5)
    axes[1].set_title("Throughput (FPS) [Higher is better]", color="white")
    axes[1].tick_params(colors="white")
    for b in b2:
        axes[1].annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontweight="bold")

    axes[2].set_facecolor("#282a36")
    bp = axes[2].boxplot([results[e]["latencies"] for e in engines], tick_labels=engines, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[2].set_title("Latency Distribution", color="white")
    axes[2].tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, facecolor=fig.get_facecolor())
    plt.close()

def save_report(results, report_path: Path, args):
    base_lat = results["PyTorch"]["stats"]["mean_ms"]
    lines = [
        "# Лабораторна робота №4: Порівняльний аналіз Inference Frameworks",
        "",
        "## 1. Конфігурація бенчмарку",
        f"- **Модель:** `{args.model_name or 'mobilenet_v3_small'}`",
        f"- **Розмір входу:** `{args.img_size}x{args.img_size}`, Batch size: `{args.batch_size}`",
        f"- **Ітерації:** `{args.warmup}` warmup / `{args.iterations}` тест",
        "",
        "## 2. Порівняльна таблиця результатів",
        "",
        "| Фреймворк / Двигун | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | P95 (ms) | Std (ms) | FPS | Speedup | RSS (MB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for eng, d in results.items():
        st = d["stats"]
        sp = base_lat / st["mean_ms"] if st["mean_ms"] > 0 else 1.0
        lines.append(f"| **{eng}** | {st['mean_ms']:.2f} | {st['median_ms']:.2f} | {st['min_ms']:.2f} | {st['max_ms']:.2f} | {st['p95_ms']:.2f} | {st['std_ms']:.2f} | **{st['fps']:.1f}** | **{sp:.2f}x** | {d['mem']:.1f} |")

    lines.extend([
        "",
        "## 3. Графіки порівняння",
        "",
        "![Benchmark Comparison](plots/benchmark_comparison.png)",
        "",
        "## 4. Пояснення отриманих відмінностей",
        "- **ONNX Runtime:** Використовує Constant Folding (згортання константних вузлів), Operator Fusion (об'єднання Conv+BN+Activation в єдиний монолітний кернел), мінімізує виділення пам'яті та усуває накладні витрати інтерпретатора Python.",
        "- **OpenVINO:** Задіює оптимізовані інструкції Intel CPU (AVX-512, AVX2, VNNI), перетворює тензори у блокову розкладку пам'яті (Blocked Layout) під ширину SIMD-регістрів і кеш-ліній та ефективно балансує навантаження через TBB (Thread Building Blocks).",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    args = parse_args()
    model, _, _, _ = load_trained_model(args.weights, args.model_name)
    inp = np.random.randn(args.batch_size, 3, args.img_size, args.img_size).astype(np.float32)

    torch_inp = torch.from_numpy(inp)
    with torch.no_grad():
        t_lat, t_mem = measure_fn(lambda: model(torch_inp), args.warmup, args.iterations)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.onnx), sess_options=opts, providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: inp}
    out_name = [session.get_outputs()[0].name]
    o_lat, o_mem = measure_fn(lambda: session.run(out_name, feed), args.warmup, args.iterations)

    core = ov.Core()
    compiled_model = core.compile_model(core.read_model(str(args.onnx)), "CPU")
    ov_out = compiled_model.output(0)
    v_lat, v_mem = measure_fn(lambda: compiled_model([inp])[ov_out], args.warmup, args.iterations)

    results = {
        "PyTorch": {"latencies": t_lat, "stats": compute_stats(t_lat, args.batch_size), "mem": t_mem},
        "ONNX Runtime": {"latencies": o_lat, "stats": compute_stats(o_lat, args.batch_size), "mem": o_mem},
        "OpenVINO": {"latencies": v_lat, "stats": compute_stats(v_lat, args.batch_size), "mem": v_mem},
    }

    table = Table(title="Benchmark Performance Comparison", border_style="bright_blue")
    table.add_column("Framework / Engine", style="bold cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("Median (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("Std (ms)", justify="right")
    table.add_column("FPS", justify="right", style="bold green")
    table.add_column("Speedup", justify="right", style="bold yellow")
    table.add_column("RSS (MB)", justify="right")

    base = results["PyTorch"]["stats"]["mean_ms"]
    for eng, d in results.items():
        st = d["stats"]
        sp = base / st["mean_ms"] if st["mean_ms"] > 0 else 1.0
        table.add_row(eng, f"{st['mean_ms']:6.2f}", f"{st['median_ms']:6.2f}", f"{st['p95_ms']:6.2f}", f"{st['std_ms']:6.2f}", f"{st['fps']:6.1f}", f"{sp:5.2f}x", f"{d['mem']:6.1f}")
    console.print(table)

    save_plots(results, args.output_dir / "benchmark_comparison.png")
    save_report(results, args.report, args)
    console.print(f"[bold green]Saved benchmark chart to:[/bold green] {args.output_dir / 'benchmark_comparison.png'}")
    console.print(f"[bold green]Saved markdown report to:[/bold green] {args.report}")

if __name__ == "__main__":
    main()
