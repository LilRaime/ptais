import argparse
import gc
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import openvino as ov
import psutil
from rich.console import Console
from rich.table import Table
from src.hardware import audit_hardware_devices, get_system_specs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-model", type=Path, default=Path("weights/mobilenet_v3_fp32.onnx"))
    parser.add_argument("--int8-model", type=Path, default=Path("weights/mobilenet_v3_int8.onnx"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--output-plot", type=Path, default=Path("plots/devices_benchmark_comparison.png"))
    return parser.parse_args()

def get_process_memory_mb():
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def run_benchmark(fn, warmup_iters, test_iters):
    for _ in range(warmup_iters):
        fn()
    mem = get_process_memory_mb()
    latencies = []
    for _ in range(test_iters):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(latencies)
    mean_lat = float(np.mean(lat))
    return {
        "mean_ms": mean_lat,
        "median_ms": float(np.median(lat)),
        "min_ms": float(np.min(lat)),
        "max_ms": float(np.max(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "std_ms": float(np.std(lat)),
        "fps": float(1000.0 / mean_lat) if mean_lat > 0 else 0.0,
        "mem_mb": mem,
        "latencies": lat,
    }

def save_devices_plot(results, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(results.keys())
    means = [results[k]["mean_ms"] for k in labels]
    stds = [results[k]["std_ms"] for k in labels]
    fps_vals = [results[k]["fps"] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    fig.patch.set_facecolor("#181920")
    palette = ["#3498db", "#2ecc71", "#9b59b6", "#e74c3c", "#f39c12", "#1abc9c"]

    for ax in axes:
        ax.set_facecolor("#22232e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#44475a")

    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=15, ha="right", color="white")
    b1 = axes[0].bar(labels, means, yerr=stds, capsize=4, color=palette[:len(labels)], alpha=0.9, width=0.45)
    axes[0].set_title("Device Latency (ms)\n[Lower is better]", color="white", fontsize=11, fontweight="bold")
    for b in b1:
        axes[0].annotate(f"{b.get_height():.2f} ms", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontweight="bold", fontsize=9)

    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right", color="white")
    b2 = axes[1].bar(labels, fps_vals, color=palette[:len(labels)], alpha=0.9, width=0.45)
    axes[1].set_title("Device Throughput (FPS)\n[Higher is better]", color="white", fontsize=11, fontweight="bold")
    for b in b2:
        axes[1].annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=160, facecolor=fig.get_facecolor())
    plt.close()

def main():
    args = parse_args()
    specs = get_system_specs()
    audit = audit_hardware_devices()

    audit_table = Table(title="Hardware Audit: Available vs Missing Accelerators", border_style="bright_blue")
    audit_table.add_column("Device Class", style="cyan")
    audit_table.add_column("Status", justify="center")
    audit_table.add_column("Hardware Specifications", style="bold")
    audit_table.add_column("Audit Notes / Absence Justification", style="dim")

    audit_table.add_row("CPU", "[bold green]AVAILABLE[/bold green]", f"{specs['cpu']} ({specs['cpu_cores_physical']}C/{specs['cpu_cores_logical']}T)", audit["CPU"]["reason"])
    if audit["GPU"]["available"]:
        audit_table.add_row("GPU", "[bold green]AVAILABLE[/bold green]", audit["GPU"]["details"] or "OpenVINO GPU target", audit["GPU"]["reason"])
    else:
        audit_table.add_row("GPU", "[bold red]ABSENT[/bold red]", "None", audit["GPU"]["reason"])

    if audit["NPU"]["available"]:
        audit_table.add_row("NPU", "[bold green]AVAILABLE[/bold green]", audit["NPU"]["details"] or "NPU", audit["NPU"]["reason"])
    else:
        audit_table.add_row("NPU", "[bold red]ABSENT[/bold red]", "None", audit["NPU"]["reason"])

    console.print(audit_table)

    inp = np.random.randn(args.batch_size, 3, args.img_size, args.img_size).astype(np.float32)
    results = {}

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_cpu_fp32 = ort.InferenceSession(str(args.fp32_model), sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = ort_cpu_fp32.get_inputs()[0].name
    out_name = [ort_cpu_fp32.get_outputs()[0].name]
    results["ORT CPU (FP32)"] = run_benchmark(lambda: ort_cpu_fp32.run(out_name, {in_name: inp}), args.warmup, args.iterations)

    ort_cpu_int8 = ort.InferenceSession(str(args.int8_model), sess_options=opts, providers=["CPUExecutionProvider"])
    results["ORT CPU (INT8)"] = run_benchmark(lambda: ort_cpu_int8.run(out_name, {in_name: inp}), args.warmup, args.iterations)

    core = ov.Core()
    ov_model_fp32 = core.read_model(str(args.fp32_model))
    ov_model_int8 = core.read_model(str(args.int8_model))

    ov_cpu_fp32 = core.compile_model(ov_model_fp32, "CPU")
    ov_out_cpu = ov_cpu_fp32.output(0)
    results["OpenVINO CPU (FP32)"] = run_benchmark(lambda: ov_cpu_fp32([inp])[ov_out_cpu], args.warmup, args.iterations)

    ov_cpu_int8 = core.compile_model(ov_model_int8, "CPU")
    ov_out_int8 = ov_cpu_int8.output(0)
    results["OpenVINO CPU (INT8)"] = run_benchmark(lambda: ov_cpu_int8([inp])[ov_out_int8], args.warmup, args.iterations)

    gpu_devices = [d for d in core.available_devices if d.startswith("GPU")]
    for g_dev in gpu_devices:
        dev_title = f"OpenVINO {g_dev}"
        try:
            full_name = core.get_property(g_dev, "FULL_DEVICE_NAME")
            if "NVIDIA" in full_name or "RTX" in full_name:
                dev_title = "OpenVINO GPU (RTX 4060)"
            elif "1035" in full_name or "Radeon" in full_name:
                dev_title = "OpenVINO GPU (Radeon iGPU)"
        except Exception:
            pass

        try:
            ov_gpu = core.compile_model(ov_model_fp32, g_dev)
            ov_out_gpu = ov_gpu.output(0)
            results[dev_title] = run_benchmark(lambda: ov_gpu([inp])[ov_out_gpu], args.warmup, args.iterations)
        except Exception as e:
            console.print(f"[yellow]Skipping unsupported GPU device {g_dev} ({dev_title}): {e}[/yellow]")

    bench_table = Table(title="Hardware & Engine Benchmark Results", border_style="bright_blue")
    bench_table.add_column("Device & Engine Target", style="cyan")
    bench_table.add_column("Mean (ms)", justify="right")
    bench_table.add_column("Median (ms)", justify="right")
    bench_table.add_column("P95 (ms)", justify="right")
    bench_table.add_column("FPS", justify="right", style="bold green")
    bench_table.add_column("Relative Speedup", justify="right", style="bold yellow")
    bench_table.add_column("Process RAM (MB)", justify="right")

    base_lat = results["ORT CPU (FP32)"]["mean_ms"]
    for target, res in results.items():
        sp = base_lat / res["mean_ms"] if res["mean_ms"] > 0 else 1.0
        bench_table.add_row(
            target,
            f"{res['mean_ms']:.2f}",
            f"{res['median_ms']:.2f}",
            f"{res['p95_ms']:.2f}",
            f"{res['fps']:.1f}",
            f"{sp:.2f}x",
            f"{res['mem_mb']:.1f}",
        )

    console.print(bench_table)

    save_devices_plot(results, args.output_plot)
    console.print(f"[bold green]Saved hardware benchmark chart to:[/bold green] {args.output_plot}")

if __name__ == "__main__":
    main()
