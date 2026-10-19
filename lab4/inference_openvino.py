import argparse
from pathlib import Path
import time
import cv2
import openvino as ov
import sys
from rich.console import Console
from rich.table import Table
from src.pipeline import ImageStream, preprocess_image, postprocess_output
from src.visualizer import draw_overlay

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    default_img = Path("../lab1/dataset/cat/cat_001.jpg") if Path("../lab1/dataset/cat/cat_001.jpg").exists() else Path("../lab1/dataset")
    parser.add_argument("--source", type=Path, default=default_img)
    parser.add_argument("--model", type=Path, default=Path("weights/mobilenet_v3_small.onnx"))
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--classes", nargs="+", default=["cat", "dog", "raven"])
    parser.add_argument("--save", nargs="?", const=Path("plots/openvino_results"), default=None, type=Path)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--delay", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    core = ov.Core()
    model = core.read_model(str(args.model))
    compiled_model = core.compile_model(model, args.device)
    output_layer = compiled_model.output(0)

    stream = ImageStream(args.source)
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    table = Table(title="OpenVINO Inference Results", border_style="bright_blue")
    table.add_column("Filename", style="cyan")
    table.add_column("Predicted Class", style="bold green")
    table.add_column("Confidence", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("FPS", justify="right")

    for frame, img_path in stream:
        tensor = preprocess_image(frame, img_size=args.img_size)
        t0 = time.perf_counter()
        results = compiled_model([tensor])[output_layer]
        lat = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / lat if lat > 0 else 0.0

        top_class, top_conf, class_probs = postprocess_output(results, args.classes)
        table.add_row(img_path.name, top_class, f"{top_conf * 100:.1f}%", f"{lat:5.2f}", f"{fps:5.1f}")

        annotated = draw_overlay(frame, top_class, top_conf, class_probs, fps, lat, "OpenVINO", args.model.stem)
        if args.save:
            cv2.imwrite(str(args.save / f"openvino_{img_path.name}"), annotated)

        if not args.no_gui:
            cv2.imshow("OpenVINO Inference", annotated)
            if cv2.waitKey(args.delay) in [27, ord('q')]:
                break

    if not args.no_gui:
        cv2.destroyAllWindows()
    console.print(table)

if __name__ == "__main__":
    main()
