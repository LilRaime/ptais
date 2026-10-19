import argparse
from pathlib import Path
import sys
import onnx
from rich.console import Console
from rich.table import Table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def extract_shape(tensor_type):
    return [dim.dim_value if dim.HasField("dim_value") else (dim.dim_param if dim.HasField("dim_param") else "?") for dim in tensor_type.shape.dim]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("weights/mobilenet_v3_small.onnx"))
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    onnx.checker.check_model(model)
    console.print(f"[bold green][OK] ONNX model {args.model} is structurally valid.[/bold green]\n")

    table = Table(title="ONNX Graph Inputs & Outputs", border_style="bright_blue")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("DType", style="yellow")
    table.add_column("Shape", style="green")

    for inp in model.graph.input:
        table.add_row("Input", inp.name, onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type), str(extract_shape(inp.type.tensor_type)))
    for out in model.graph.output:
        table.add_row("Output", out.name, onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type), str(extract_shape(out.type.tensor_type)))
    console.print(table)

    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    t_ops = Table(title="Top Operators in Graph", border_style="blue")
    t_ops.add_column("Op Type", style="cyan")
    t_ops.add_column("Count", justify="right", style="yellow")
    for op, cnt in sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:8]:
        t_ops.add_row(op, str(cnt))
    console.print(t_ops)

if __name__ == "__main__":
    main()
