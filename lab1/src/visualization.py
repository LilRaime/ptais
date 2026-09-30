from collections import Counter
from pathlib import Path
from typing import Sequence
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .augmentation import augment_image

ImageSample = tuple[str | Path, str]

PALETTE = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6"]


def set_plot_theme() -> None:
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#E2E8F0"
    plt.rcParams["axes.linewidth"] = 0.8


def show_class_distribution(
    images: Sequence[ImageSample],
    title: str = "Class Distribution in Dataset",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    set_plot_theme()
    counts_map = Counter(class_name for _, class_name in images)
    classes = sorted(counts_map.keys())
    counts = [counts_map[c] for c in classes]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=120)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(classes))]
    bars = ax.bar(classes, counts, color=bar_colors, width=0.55, edgecolor="none", zorder=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0", zorder=0)

    for bar in bars:
        height = bar.get_height()
        pct = (height / total * 100.0) if total > 0 else 0
        ax.annotate(
            f"{int(height)}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#334155",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=16, color="#0F172A")
    ax.set_xlabel("Class", fontsize=11, fontweight="600", labelpad=8, color="#475569")
    ax.set_ylabel("Number of Images", fontsize=11, fontweight="600", labelpad=8, color="#475569")
    ax.tick_params(colors="#475569", labelsize=10)

    if counts:
        ax.set_ylim(0, max(counts) * 1.25)

    plt.tight_layout()
    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def show_class_examples(
    images: Sequence[ImageSample],
    examples_per_class: int = 5,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    set_plot_theme()
    classes = sorted({class_name for _, class_name in images})

    for class_name in classes:
        class_images = [str(p) for p, c in images if c == class_name][:examples_per_class]
        if not class_images:
            continue

        fig, axes = plt.subplots(1, len(class_images), figsize=(3 * len(class_images), 3.2), dpi=120)
        fig.patch.set_facecolor("#FAFAFA")
        if len(class_images) == 1:
            axes = [axes]

        for idx, (ax, path) in enumerate(zip(axes, class_images)):
            image = cv2.imread(path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ax.imshow(image_rgb)
            ax.set_title(f"Sample #{idx + 1}", fontsize=10, color="#475569", pad=6)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.suptitle(f"Class Showcase: {class_name.capitalize()}", fontsize=13, fontweight="bold", color="#0F172A", y=1.02)
        plt.tight_layout()

        if save_path:
            p = Path(save_path)
            out_file = p.parent / f"{p.stem}_{class_name.lower()}{p.suffix}"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_file, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


def show_augmentations(
    path: str | Path,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    set_plot_theme()
    image = cv2.imread(str(path))
    if image is None:
        print(f"[!] Error: Could not read image for augmentation display: {path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    flipped, rotated, brighter = augment_image(image_rgb)

    previews = [
        ("Original", image_rgb),
        ("Horizontal Flip", flipped),
        ("Rotation (15°)", rotated),
        ("Brightness (x1.3)", brighter),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=120)
    fig.patch.set_facecolor("#FAFAFA")

    for ax, (label, img) in zip(axes, previews):
        ax.imshow(img)
        ax.set_title(label, fontsize=11, fontweight="bold", color="#1E293B", pad=8)
        ax.axis("off")

    fig.suptitle(f"Data Augmentation Pipeline Preview: {Path(path).name}", fontsize=13, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)