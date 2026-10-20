import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.pipeline import (
    EndToEndPipeline,
    ImageStream,
    OpenVinoEngine,
    OrtEngine,
    analyze_bottleneck,
)
from src.visualizer import draw_overlay

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../lab1/dataset"))
    parser.add_argument("--model", type=Path, default=Path("weights/mobilenet_v3_int8.onnx"))
    parser.add_argument("--engine", type=str, choices=["ort", "openvino"], default="ort")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--classes", nargs="+", default=["cat", "dog", "raven"])
    parser.add_argument("--save", type=Path, default=Path("plots/pipeline_results"))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--max-items", type=int, default=30)
    parser.add_argument("--plot-breakdown", type=Path, default=Path("plots/pipeline_stage_breakdown.png"))
    return parser.parse_args()

def save_stage_breakdown_plot(mean_pre, mean_inf, mean_post, total, bottleneck_stage, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    fig.patch.set_facecolor("#181920")

    for ax in axes:
        ax.set_facecolor("#22232e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#44475a")

    stages = ["Preprocessing", "Inference", "Postprocessing"]
    times = [mean_pre, mean_inf, mean_post]
    colors = ["#f39c12", "#e74c3c" if bottleneck_stage == "Inference" else "#3498db", "#2ecc71"]

    b1 = axes[0].bar(stages, times, color=colors, alpha=0.9, width=0.45)
    axes[0].set_title("Stage Latency (ms)", color="white", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Duration (ms)", color="white")
    for b in b1:
        axes[0].annotate(f"{b.get_height():.2f} ms", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", color="white", fontweight="bold")

    pcts = [(t / total) * 100.0 for t in times]
    wedges, texts, autotexts = axes[1].pie(
        pcts,
        labels=stages,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        textprops={"color": "white"},
        wedgeprops={"edgecolor": "#181920", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontweight("bold")
    axes[1].set_title(f"Time Share (%) - Bottleneck: {bottleneck_stage}", color="white", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=160, facecolor=fig.get_facecolor())
    plt.close()

def main():
    args = parse_args()

    if args.engine == "ort":
        engine = OrtEngine(args.model)
        engine_name = "ONNX Runtime"
    else:
        engine = OpenVinoEngine(args.model)
        engine_name = "OpenVINO"

    pipeline = EndToEndPipeline(engine, args.classes, args.img_size)
    stream = ImageStream(args.source)

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    table = Table(title="Integrated Pipeline Execution Logs", border_style="bright_blue")
    table.add_column("Filename", style="cyan")
    table.add_column("Prediction", style="bold green")
    table.add_column("Confidence", justify="right")
    table.add_column("Pre (ms)", justify="right")
    table.add_column("Inf (ms)", justify="right")
    table.add_column("Post (ms)", justify="right")
    table.add_column("Total (ms)", justify="right", style="bold yellow")
    table.add_column("FPS", justify="right")

    pre_times = []
    inf_times = []
    post_times = []
    total_times = []

    count = 0
    for frame, path in stream:
        top_cls, conf, probs, timing = pipeline.process_frame(frame)
        pre_times.append(timing["preprocess_ms"])
        inf_times.append(timing["inference_ms"])
        post_times.append(timing["postprocess_ms"])
        total_times.append(timing["total_ms"])

        fps = 1000.0 / timing["total_ms"] if timing["total_ms"] > 0 else 0.0
        table.add_row(
            path.name,
            top_cls,
            f"{conf * 100:.1f}%",
            f"{timing['preprocess_ms']:.2f}",
            f"{timing['inference_ms']:.2f}",
            f"{timing['postprocess_ms']:.2f}",
            f"{timing['total_ms']:.2f}",
            f"{fps:.1f}",
        )

        annotated = draw_overlay(frame, top_cls, conf, probs, fps, timing, engine_name, args.model.stem)
        if args.save:
            cv2.imwrite(str(args.save / f"out_{path.name}"), annotated)

        if args.gui:
            cv2.imshow("Pipeline HUD", annotated)
            if cv2.waitKey(1) in [27, ord("q")]:
                break

        count += 1
        if count >= args.max_items:
            break

    if args.gui:
        cv2.destroyAllWindows()

    console.print(table)

    mean_pre = float(np.mean(pre_times))
    mean_inf = float(np.mean(inf_times))
    mean_post = float(np.mean(post_times))
    mean_tot = float(np.mean(total_times))

    stage_stats = {"preprocess_ms": mean_pre, "inference_ms": mean_inf, "postprocess_ms": mean_post}
    bottleneck_info = analyze_bottleneck(stage_stats)

    b_table = Table(title="Pipeline Stage Timing Breakdown & Profiling", border_style="magenta")
    b_table.add_column("Pipeline Stage", style="cyan")
    b_table.add_column("Mean Time (ms)", justify="right")
    b_table.add_column("Time Share (%)", justify="right")
    b_table.add_column("Role / Function", style="dim")

    b_table.add_row("1. Preprocessing", f"{mean_pre:.2f} ms", f"{bottleneck_info['percentages']['preprocess']:.1f}%", "BGR2RGB, Resize(224x224), Normalization, CHW")
    b_table.add_row("2. Inference", f"{mean_inf:.2f} ms", f"{bottleneck_info['percentages']['inference']:.1f}%", f"{engine_name} execution on {args.model.name}")
    b_table.add_row("3. Postprocessing", f"{mean_post:.2f} ms", f"{bottleneck_info['percentages']['postprocess']:.1f}%", "Softmax logits, Argmax, Probability mapping")
    b_table.add_row("Total Pipeline Latency", f"{mean_tot:.2f} ms", "100.0%", f"End-to-end throughput: {1000.0 / mean_tot:.1f} FPS")
    console.print(b_table)

    diagnosis_text = (
        f"[bold red]Identified Bottleneck: {bottleneck_info['bottleneck_stage']}[/bold red]\n"
        f"Diagnosis: {bottleneck_info['diagnosis']}\n\n"
        "[bold cyan]Actionable Recommendations:[/bold cyan]\n"
        + "\n".join([f"- {r}" for r in bottleneck_info["recommendations"]])
    )
    console.print(Panel(diagnosis_text, title="Bottleneck Analysis & Optimization Strategy", border_style="yellow"))

    save_stage_breakdown_plot(mean_pre, mean_inf, mean_post, mean_tot, bottleneck_info["bottleneck_stage"], args.plot_breakdown)
    console.print(f"[bold green]Saved stage breakdown plot to:[/bold green] {args.plot_breakdown}")

if __name__ == "__main__":
    main()
