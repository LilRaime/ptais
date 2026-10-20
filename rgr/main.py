import argparse
import os
import subprocess
import sys

from rich.console import Console

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def run_step(step_num: int, title: str, cmd: list):
    console.print("\n[bold cyan]==================================================[/bold cyan]")
    console.print(f"[bold cyan]Step {step_num}: {title}[/bold cyan]")
    console.print("[bold cyan]==================================================[/bold cyan]")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    res = subprocess.run([sys.executable] + cmd, env=env)
    if res.returncode != 0:
        console.print(f"[bold red]Step {step_num} failed with code {res.returncode}[/bold red]")
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--pipeline-items", type=int, default=15)
    args = parser.parse_args()

    console.print("\n[bold green]Starting PTAIS RGR Optimization & Benchmarking Pipeline[/bold green]")

    run_step(1, "Export FP32 Base ONNX Model", ["export_fp32.py"])
    run_step(2, "Quantize Model to INT8 (Dynamic & Static)", ["quantize_model.py"])
    run_step(3, "Compare Base vs Quantized Model (Accuracy, Size, Latency, FPS, RAM)", ["compare_models.py", "--iterations", str(args.iterations), "--warmup", str(args.warmup)])
    run_step(4, "Execute Integrated Pipeline & Profile Bottleneck (Pre vs Inf vs Post)", ["run_pipeline.py", "--max-items", str(args.pipeline_items)])
    run_step(5, "Benchmark Execution Devices (CPU / GPU / NPU Audit)", ["benchmark_devices.py", "--iterations", str(args.iterations), "--warmup", str(args.warmup)])
    run_step(6, "Compile Final Report & Decision Matrix", ["generate_report.py"])

    console.print("\n[bold green][SUCCESS] All tasks completed successfully! Review benchmark_report.md and plots/ directory.[/bold green]\n")

if __name__ == "__main__":
    main()
