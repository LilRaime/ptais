from collections import Counter
from pathlib import Path
from typing import NamedTuple, Sequence
import cv2

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

ImageSample = tuple[str, str]


class DatasetAnalysisResult(NamedTuple):
    classes: list[str]
    images: list[ImageSample]
    bad_files: list[str]


def analyze_dataset(
    dataset_dir: str | Path,
    valid_extensions: Sequence[str] = SUPPORTED_EXTENSIONS,
    verbose: bool = True,
) -> DatasetAnalysisResult:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    normalized_exts = tuple(ext.lower() for ext in valid_extensions)

    classes: list[str] = []
    images: list[ImageSample] = []

    for class_path in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_path.name
        classes.append(class_name)

        for file_path in sorted(class_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in normalized_exts:
                images.append((str(file_path), class_name))

    bad_files: list[str] = []
    sizes: list[tuple[int, int]] = []

    for path_str, _ in images:
        img = cv2.imread(path_str)
        if img is None:
            bad_files.append(path_str)
            continue
        h, w = img.shape[:2]
        sizes.append((w, h))

    if verbose:
        total_images = len(images)
        valid_images = total_images - len(bad_files)

        if HAS_RICH and console:
            summary_table = Table(title="Dataset Analysis Summary", border_style="cyan")
            summary_table.add_column("Metric", style="bold white")
            summary_table.add_column("Value", style="bold green", justify="right")

            summary_table.add_row("Target Directory", str(root.name))
            summary_table.add_row("Total Classes", str(len(classes)))
            summary_table.add_row("Total Images Found", str(total_images))
            summary_table.add_row("Valid / Corrupted", f"{valid_images} / {len(bad_files)}")
            console.print(summary_table)

            class_table = Table(title="Class Distribution", border_style="bright_blue")
            class_table.add_column("Class Name", style="cyan")
            class_table.add_column("Count", justify="right", style="magenta")
            class_table.add_column("Share", justify="right", style="yellow")

            class_counts = Counter(cls for _, cls in images)
            for cls in classes:
                cnt = class_counts.get(cls, 0)
                pct = (cnt / total_images * 100.0) if total_images > 0 else 0.0
                class_table.add_row(cls, str(cnt), f"{pct:.1f}%")
            console.print(class_table)

            if sizes:
                res_table = Table(title="Top Image Resolutions", border_style="dim")
                res_table.add_column("Resolution (W x H)", style="white")
                res_table.add_column("Count", justify="right", style="green")
                res_table.add_column("Share", justify="right", style="yellow")

                for (w, h), cnt in Counter(sizes).most_common(5):
                    pct = cnt / len(sizes) * 100.0
                    res_table.add_row(f"{w}x{h} px", str(cnt), f"{pct:.1f}%")
                console.print(res_table)

            if bad_files:
                console.print(f"[bold red]Warning: {len(bad_files)} unreadable file(s) found:[/bold red]")
                for bf in bad_files:
                    console.print(f"  [red]*[/red] {bf}")
        else:
            print("=" * 55)
            print(" DATASET ANALYSIS REPORT")
            print("=" * 55)
            print(f"Directory:       {root.name}")
            print(f"Total Classes:   {len(classes)}")
            print(f"Total Images:    {total_images}")
            print(f"Valid / Corrupt: {valid_images} / {len(bad_files)}")
            print("-" * 55)
            print(f"{'Class':<25} {'Count':>10} {'Share':>12}")
            print("-" * 55)
            class_counts = Counter(cls for _, cls in images)
            for cls in classes:
                cnt = class_counts.get(cls, 0)
                pct = (cnt / total_images * 100.0) if total_images > 0 else 0.0
                print(f"{cls:<25} {cnt:>10} {pct:>11.1f}%")
            print("=" * 55)

    return DatasetAnalysisResult(classes=classes, images=images, bad_files=bad_files)