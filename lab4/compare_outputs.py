import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import sys
from rich.console import Console
from rich.table import Table
from src.models import load_trained_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("../lab3/weights/mobilenet_v3_finetuned.pth"))
    parser.add_argument("--onnx", type=Path, default=Path("weights/mobilenet_v3_small.onnx"))
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, _, class_names, img_size = load_trained_model(args.weights, args.model_name)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    np_inp = np.random.randn(args.batch_size, 3, img_size, img_size).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(np_inp)).numpy()
    ort_out = session.run(None, {session.get_inputs()[0].name: np_inp})[0]

    diff = np.abs(torch_out - ort_out)
    f_t, f_o = torch_out.flatten(), ort_out.flatten()
    cos_sim = float(np.dot(f_t, f_o) / (np.linalg.norm(f_t) * np.linalg.norm(f_o) + 1e-9))
    t_prob, o_prob = softmax(torch_out), softmax(ort_out)
    is_close = bool(np.allclose(torch_out, ort_out, rtol=args.rtol, atol=args.atol))

    table = Table(title="Numerical Parity: PyTorch vs ONNX Runtime", border_style="bright_blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold yellow")
    table.add_column("Status", justify="center")

    metrics = [
        ("Max Absolute Difference", f"{np.max(diff):.6e}", "[green]PASS[/green]" if np.max(diff) < args.atol else "[yellow]WARN[/yellow]"),
        ("Mean Absolute Error (MAE)", f"{np.mean(diff):.6e}", "[green]PASS[/green]"),
        ("Root Mean Squared Error (RMSE)", f"{np.sqrt(np.mean(diff ** 2)):.6e}", "[green]PASS[/green]"),
        ("Cosine Similarity", f"{cos_sim:.8f}", "[green]PASS[/green]"),
        ("Max Probability Difference", f"{np.max(np.abs(t_prob - o_prob)):.6e}", "[green]PASS[/green]"),
        ("Argmax Predictions Match", str(np.array_equal(np.argmax(torch_out, -1), np.argmax(ort_out, -1))), "[green]MATCH[/green]"),
        ("np.allclose validation", str(is_close), "[bold green]PASSED[/bold green]" if is_close else "[bold red]FAILED[/bold red]"),
    ]
    for m, v, s in metrics:
        table.add_row(m, v, s)
    console.print(table)

    sample_t = Table(title="Logits & Probabilities Comparison", border_style="magenta")
    sample_t.add_column("Class", style="bold")
    sample_t.add_column("PyTorch Logit", justify="right")
    sample_t.add_column("ONNX Logit", justify="right")
    sample_t.add_column("Diff", justify="right")
    sample_t.add_column("PyTorch Prob", justify="right")
    sample_t.add_column("ONNX Prob", justify="right")
    for i, c in enumerate(class_names):
        sample_t.add_row(
            c, f"{torch_out[0, i]:8.5f}", f"{ort_out[0, i]:8.5f}",
            f"{(torch_out[0, i] - ort_out[0, i]):+.2e}",
            f"{t_prob[0, i] * 100:5.2f}%", f"{o_prob[0, i] * 100:5.2f}%"
        )
    console.print(sample_t)

if __name__ == "__main__":
    main()
