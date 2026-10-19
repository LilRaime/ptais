import argparse
import os
from pathlib import Path
import sys
import torch
from rich.console import Console
from src.models import load_trained_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    default_weights = Path("../lab3/weights/mobilenet_v3_finetuned.pth") if Path("../lab3/weights/mobilenet_v3_finetuned.pth").exists() else Path("../lab2/best_model.pth")
    parser.add_argument("--weights", type=Path, default=default_weights)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()

def main():
    args = parse_args()
    model, model_name, _, img_size = load_trained_model(args.weights, args.model_name)
    img_size = args.img_size or img_size
    output = args.output or Path(f"weights/{model_name}.onnx")
    output.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(args.batch_size, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(output),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    console.print(f"[bold green]Exported ONNX model to:[/bold green] {output} ({output.stat().st_size / (1024 * 1024):.2f} MB)")

if __name__ == "__main__":
    main()
