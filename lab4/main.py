import argparse
import os
import subprocess
import sys
from rich.console import Console

console = Console()

def run_step(step_num: int, title: str, cmd: list):
    console.print(f"\n[bold cyan]Step {step_num}: {title}[/bold cyan]")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    res = subprocess.run([sys.executable] + cmd, env=env)
    if res.returncode != 0:
        console.print(f"[bold red]Step {step_num} failed with code {res.returncode}[/bold red]")
        sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    run_step(1, "Export PyTorch Model to ONNX", ["export_onnx.py"])
    run_step(2, "Verify ONNX Model Validity", ["verify_onnx.py"])
    run_step(3, "Compare Outputs & Calculate Divergence", ["compare_outputs.py"])

    if not args.skip_inference:
        run_step(4, "Standalone ONNX Runtime Inference", ["inference_onnx.py", "--no-gui", "--save", "plots/onnx_results"])
        run_step(5, "OpenVINO Native ONNX Inference", ["inference_openvino.py", "--no-gui", "--save", "plots/openvino_results"])

    run_step(6, "Benchmark PyTorch vs ONNX Runtime vs OpenVINO", ["benchmark.py", "--iterations", str(args.iterations), "--warmup", "20"])
    console.print("\n[bold green][OK] All Lab 4 Tasks Completed Successfully![/bold green]")

if __name__ == "__main__":
    main()
