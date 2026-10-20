import argparse
import sys
from pathlib import Path

import onnx
import torch
from rich.console import Console
from rich.table import Table
from src.models import load_trained_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    default_weights = Path("../lab3/weights/mobilenet_v3_finetuned.pth")
    if not default_weights.exists():
        default_weights = Path("../lab2/best_model.pth")
    parser.add_argument("--weights", type=Path, default=default_weights)
    parser.add_argument("--model-name", type=str, default="mobilenet_v3_small")
    parser.add_argument("--output", type=Path, default=Path("weights/mobilenet_v3_fp32.onnx"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()

def main():
    args = parse_args()
    model, model_name, class_names, img_size = load_trained_model(args.weights, args.model_name)
    img_size = args.img_size or img_size
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(args.batch_size, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy_input,
        str(args.output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    onnx_model = onnx.load(str(args.output))
    onnx.checker.check_model(onnx_model)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    table = Table(title="FP32 Base Model Specification", border_style="bright_blue")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="bold green")
    table.add_row("Model Architecture", model_name)
    table.add_row("Classes", ", ".join(class_names))
    table.add_row("Input Tensor", f"[batch_size, 3, {img_size}, {img_size}] (FP32)")
    table.add_row("Output Tensor", f"[batch_size, {len(class_names)}] (FP32 Logits)")
    table.add_row("File Size", f"{size_mb:.2f} MB")
    table.add_row("Export Path", str(args.output))
    table.add_row("Graph Verification", "PASSED (onnx.checker)")
    console.print(table)

if __name__ == "__main__":
    main()
