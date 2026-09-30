import argparse
from pathlib import Path
import sys

from src.data_loader import load_dataset
from src.dataset_analysis import analyze_dataset
from src.split import split_dataset
from src.visualization import (
    show_augmentations,
    show_class_distribution,
    show_class_examples,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_DIR = Path("dataset")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def print_banner() -> None:
    if HAS_RICH and console:
        console.print(
            Panel.fit(
                "[bold cyan]Computer Vision Pipeline[/bold cyan]\n"
                "[dim]Dataset Analysis | Augmentation | Stratified Split | Tensor Loader[/dim]",
                border_style="bright_blue",
                padding=(1, 3),
            )
        )
    else:
        print("=" * 60)
        print("  COMPUTER VISION PIPELINE")
        print("  Dataset Analysis | Augmentation | Split | Tensor Loader")
        print("=" * 60)


def display_results(
    x_train, y_train,
    x_val, y_val,
    x_test, y_test,
    class_names: list[str],
) -> None:
    if HAS_RICH and console:
        table = Table(title="Processed Dataset Tensors", border_style="bright_blue", show_header=True)
        table.add_column("Subset", style="bold cyan", width=14)
        table.add_column("Features Shape (X)", style="green", width=22)
        table.add_column("Labels Shape (y)", style="yellow", width=18)
        table.add_column("Data Type", style="magenta", width=12)

        table.add_row("Train", str(x_train.shape), str(y_train.shape), str(x_train.dtype))
        table.add_row("Validation", str(x_val.shape), str(y_val.shape), str(x_val.dtype))
        table.add_row("Test", str(x_test.shape), str(y_test.shape), str(x_test.dtype))
        console.print("\n", table)

        class_table = Table(title="Class Index Encoding", border_style="cyan")
        class_table.add_column("Index", style="bold yellow", justify="center", width=8)
        class_table.add_column("Class Label", style="bold white", width=20)
        for idx, name in enumerate(class_names):
            class_table.add_row(str(idx), name)
        console.print(class_table)
    else:
        print("\n" + "=" * 50)
        print("PROCESSED DATASET SUMMARY")
        print("=" * 50)
        print(f"X_train: {x_train.shape} | y_train: {y_train.shape}")
        print(f"X_val:   {x_val.shape}   | y_val:   {y_val.shape}")
        print(f"X_test:  {x_test.shape}  | y_test:  {y_test.shape}")
        print("\nClass Mapping:")
        for idx, name in enumerate(class_names):
            print(f"  [{idx}] -> {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Dataset Processing & Splitting Pipeline")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Path to dataset root folder")
    parser.add_argument("--skip-viz", action="store_true", help="Skip interactive matplotlib visual displays")
    parser.add_argument("--save-plots", action="store_true", help="Save visual plots to 'plots/' directory")
    args = parser.parse_args()

    print_banner()

    classes, images, bad_files = analyze_dataset(args.dataset_dir)
    images = [(path, class_name) for path, class_name in images if path not in bad_files]

    if not images:
        if HAS_RICH and console:
            console.print("[bold red]Error:[/bold red] The dataset is empty or contains no valid images.")
        else:
            print("Error: The dataset is empty or contains no valid images.")
        sys.exit(1)

    aug_save = Path("plots/augmentation_preview.png") if args.save_plots else None
    if not args.skip_viz or args.save_plots:
        print("\n[+] Preparing data augmentation preview...")
        show_augmentations(images[0][0], save_path=aug_save, show=not args.skip_viz)

    train, validation, test = split_dataset(images)

    dist_save = Path("plots/class_distribution.png") if args.save_plots else None
    examples_save = Path("plots/class_examples.png") if args.save_plots else None
    if not args.skip_viz or args.save_plots:
        print("[+] Preparing visual analytics...")
        show_class_distribution(images, title="Class Distribution in Dataset", save_path=dist_save, show=not args.skip_viz)
        show_class_examples(images, examples_per_class=5, save_path=examples_save, show=not args.skip_viz)

    print("\n[+] Loading and preprocessing tensors...")
    x_train, y_train, class_names = load_dataset(train)
    x_val, y_val, _ = load_dataset(validation)
    x_test, y_test, _ = load_dataset(test)

    display_results(x_train, y_train, x_val, y_val, x_test, y_test, class_names)


if __name__ == "__main__":
    main()