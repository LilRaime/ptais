import argparse
from pathlib import Path
import sys
import time
import cv2
import torch
from rich.console import Console
from rich.table import Table

from src.models import get_model
from src.pipeline import ImageStream, preprocess_frame, postprocess_output
from src.visualizer import draw_overlay

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    default_img = Path("../lab1/dataset/cat/cat_001.jpg") if Path("../lab1/dataset/cat/cat_001.jpg").exists() else Path("../lab1/dataset")
    parser.add_argument("--source", type=Path, default=default_img)
    parser.add_argument("--weights", type=Path, default=Path("weights/mobilenet_v3_finetuned.pth"))
    parser.add_argument("--model-name", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18", "simple_cnn"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", nargs="?", const=Path("plots/inference_results"), default=None, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--delay", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    class_names = ["cat", "dog", "raven"]
    model = get_model(name=args.model_name, num_classes=len(class_names), pretrained=False, freeze_backbone=False)

    if args.weights.exists():
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "class_names" in checkpoint:
                class_names = checkpoint["class_names"]
            if "img_size" in checkpoint:
                args.img_size = checkpoint["img_size"]
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        console.print(f"Loaded weights for [bold]{args.model_name}[/bold] from {args.weights}")
    else:
        console.print(f"[yellow]Warning: Weights {args.weights} not found. Running with initial weights.[/yellow]")

    model.to(device).eval()

    stream = ImageStream(args.source)
    console.print(f"Loaded [bold green]{len(stream)}[/bold green] image(s) from: {args.source}")

    save_dir = args.save if args.save is not None else args.output
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    table = Table(title="Image Inference Results", border_style="bright_blue")
    table.add_column("Filename", style="cyan")
    table.add_column("Predicted Class", style="bold green")
    table.add_column("Confidence", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("FPS", justify="right")

    for idx, (frame, img_path) in enumerate(stream, start=1):
        tensor = preprocess_frame(frame, img_size=(args.img_size, args.img_size), device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(tensor)
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        top_class, confidence, class_probs = postprocess_output(outputs, class_names)

        annotated = draw_overlay(
            frame=frame,
            top_class=top_class,
            confidence=confidence,
            class_probs=class_probs,
            fps=fps,
            latency_ms=latency_ms,
            model_name=args.model_name
        )

        table.add_row(
            img_path.name,
            top_class,
            f"{confidence * 100:.1f}%",
            f"{latency_ms:.2f}",
            f"{fps:.1f}"
        )

        if save_dir is not None:
            out_file = save_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_file), annotated)

        if not args.no_gui:
            cv2.imshow("OpenCV Image Inference (Lab 3)", annotated)
            key = cv2.waitKey(args.delay) & 0xFF
            if key in (27, ord("q")):
                break

    if not args.no_gui:
        cv2.destroyAllWindows()

    console.print(table)
    if save_dir is not None:
        console.print(f"Saved annotated images to: [bold]{save_dir}[/bold]")

if __name__ == "__main__":
    main()
