import argparse
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from src.quantization import quantize_dynamic_onnx, quantize_static_onnx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-model", type=Path, default=Path("weights/mobilenet_v3_fp32.onnx"))
    parser.add_argument("--output-selective", type=Path, default=Path("weights/mobilenet_v3_int8_selective.onnx"))
    parser.add_argument("--output-full", type=Path, default=Path("weights/mobilenet_v3_int8_full.onnx"))
    parser.add_argument("--output-static", type=Path, default=Path("weights/mobilenet_v3_int8_static.onnx"))
    parser.add_argument("--primary-int8", type=Path, default=Path("weights/mobilenet_v3_int8.onnx"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("../lab1/dataset"))
    return parser.parse_args()

def main():
    args = parse_args()
    if not args.fp32_model.exists():
        console.print(f"[bold red]Error: FP32 model not found at {args.fp32_model}. Run export_fp32.py first.[/bold red]")
        sys.exit(1)

    fp32_size = args.fp32_model.stat().st_size / (1024 * 1024)

    quantize_dynamic_onnx(args.fp32_model, args.output_selective, op_types_to_quantize=["MatMul", "Gemm"], per_channel=True)
    sel_size = args.output_selective.stat().st_size / (1024 * 1024)
    sel_red = (1.0 - (sel_size / fp32_size)) * 100.0

    quantize_dynamic_onnx(args.fp32_model, args.output_full, op_types_to_quantize=None, per_channel=True)
    full_size = args.output_full.stat().st_size / (1024 * 1024)
    full_red = (1.0 - (full_size / fp32_size)) * 100.0

    static_done = False
    stat_size = 0.0
    stat_red = 0.0
    if args.dataset_dir.exists():
        try:
            quantize_static_onnx(args.fp32_model, args.output_static, args.dataset_dir)
            stat_size = args.output_static.stat().st_size / (1024 * 1024)
            stat_red = (1.0 - (stat_size / fp32_size)) * 100.0
            static_done = True
        except Exception as e:
            console.print(f"[yellow]Warning: Static calibration skipped: {e}[/yellow]")

    shutil.copyfile(args.output_selective, args.primary_int8)

    table = Table(title="Model Quantization & Compression Summary", border_style="bright_blue")
    table.add_column("Configuration", style="cyan")
    table.add_column("Precision & Method", justify="center")
    table.add_column("File Size (MB)", justify="right")
    table.add_column("Compression Ratio", justify="right", style="bold green")
    table.add_column("Status", justify="center")

    table.add_row("Base Model", "FP32 (Original)", f"{fp32_size:.2f} MB", "1.00x (Baseline)", "[green]READY[/green]")
    table.add_row("Selective INT8 (High Accuracy)", "INT8 (Linear / Gemm)", f"{sel_size:.2f} MB", f"{fp32_size / sel_size:.2f}x (-{sel_red:.1f}%)", "[bold green]PRIMARY DEPLOYED[/bold green]")
    table.add_row("Full Dynamic INT8", "INT8 (All Layers)", f"{full_size:.2f} MB", f"{fp32_size / full_size:.2f}x (-{full_red:.1f}%)", "[green]QUANTIZED[/green]")
    if static_done:
        table.add_row("Static Calibrated INT8", "INT8 (QDQ Calibrated)", f"{stat_size:.2f} MB", f"{fp32_size / stat_size:.2f}x (-{stat_red:.1f}%)", "[green]QUANTIZED[/green]")

    console.print(table)
    console.print(f"[bold green]Primary INT8 model deployed to:[/bold green] {args.primary_int8}")

if __name__ == "__main__":
    main()
