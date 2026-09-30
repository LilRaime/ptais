import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random
import subprocess
import sys
import urllib.request

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_IMAGES_PER_CLASS: int = 40
DEFAULT_SEED: int = 42

DEFAULT_CLASSES: dict[str, str] = {
    "Cat": "cat",
    "Dog": "dog",
    "Raven": "raven",
}

DATASET_DIR: Path = Path("dataset")
METADATA_DIR: Path = Path("open_images_metadata")

CLASS_DESCRIPTIONS_URL: str = (
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
)
VALIDATION_LABELS_URL: str = (
    "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv"
)
DOWNLOADER_URL: str = (
    "https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
)


def download_file(url: str, dest_path: Path) -> None:
    if dest_path.is_file():
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_RICH and console:
        console.print(f"[cyan]Downloading:[/cyan] {url}")
    else:
        print(f"Downloading: {url}")

    urllib.request.urlretrieve(url, str(dest_path))

    if HAS_RICH and console:
        console.print(f"[green]Saved to:[/green] {dest_path}")
    else:
        print(f"Saved to: {dest_path}")


def ensure_directories(classes: dict[str, str], dataset_dir: Path, metadata_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in classes.values():
        (dataset_dir / folder_name).mkdir(parents=True, exist_ok=True)


def download_metadata(metadata_dir: Path) -> tuple[Path, Path, Path]:
    descriptions_path = metadata_dir / "oidv7-class-descriptions.csv"
    labels_path = metadata_dir / "oidv7-val-annotations-human-imagelabels.csv"
    downloader_path = metadata_dir / "downloader.py"

    download_file(CLASS_DESCRIPTIONS_URL, descriptions_path)
    download_file(VALIDATION_LABELS_URL, labels_path)
    download_file(DOWNLOADER_URL, downloader_path)

    return descriptions_path, labels_path, downloader_path


def resolve_class_ids(descriptions_path: Path, target_classes: dict[str, str]) -> dict[str, str]:
    class_ids: dict[str, str] = {}
    target_lookup = {name.lower(): name for name in target_classes}

    with open(descriptions_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            entity_id, label = row[0], row[1].strip()
            label_lower = label.lower()
            if label_lower in target_lookup:
                class_ids[target_lookup[label_lower]] = entity_id

    return class_ids


def collect_image_ids(labels_path: Path, class_ids: dict[str, str]) -> dict[str, list[str]]:
    id_to_class = {cid: cname for cname, cid in class_ids.items()}
    collected_ids: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("LabelName")
            img_id = row.get("ImageID")
            if lbl in id_to_class and img_id and img_id not in seen[lbl]:
                cname = id_to_class[lbl]
                seen[lbl].add(img_id)
                collected_ids[cname].append(img_id)

    return collected_ids


def download_and_rename_images(
    downloader_script: Path,
    selected_ids: dict[str, list[str]],
    classes: dict[str, str],
    dataset_dir: Path,
    metadata_dir: Path,
    num_processes: int = 5,
) -> None:
    for class_name, folder_name in classes.items():
        class_dir = dataset_dir / folder_name
        class_list_file = metadata_dir / f"{folder_name}_images.txt"

        ids = selected_ids.get(class_name, [])
        with open(class_list_file, "w", encoding="utf-8") as f:
            for img_id in ids:
                f.write(f"validation/{img_id}\n")

        if HAS_RICH and console:
            console.print(f"\n[bold cyan]Downloading images for class:[/bold cyan] [yellow]{class_name}[/yellow]")
        else:
            print(f"\nDownloading images for class: {class_name}")

        cmd = [
            sys.executable,
            str(downloader_script),
            str(class_list_file),
            "--download_folder",
            str(class_dir),
            "--num_processes",
            str(num_processes),
        ]
        subprocess.run(cmd, check=True)

        files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() == ".jpg")
        for idx, file_path in enumerate(files, start=1):
            target_path = class_dir / f"{folder_name}_{idx:03d}.jpg"
            if file_path != target_path:
                file_path.rename(target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download subset of Open Images V7")
    parser.add_argument("--count", type=int, default=DEFAULT_IMAGES_PER_CLASS, help="Images per class")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sampling")
    parser.add_argument("--dest", type=Path, default=DATASET_DIR, help="Destination dataset directory")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if HAS_RICH and console:
        console.print(
            Panel.fit(
                "[bold green]Open Images V7 Downloader[/bold green]\n"
                f"[dim]Sampling {args.count} images per class | Seed {args.seed}[/dim]",
                border_style="cyan",
            )
        )
    else:
        print("=" * 55)
        print(" OPEN IMAGES V7 DATASET DOWNLOADER")
        print(f" Target: {args.count} imgs/class | Seed: {args.seed}")
        print("=" * 55)

    ensure_directories(DEFAULT_CLASSES, args.dest, METADATA_DIR)
    descriptions_path, labels_path, downloader_path = download_metadata(METADATA_DIR)

    class_ids = resolve_class_ids(descriptions_path, DEFAULT_CLASSES)
    for cname in DEFAULT_CLASSES:
        if cname not in class_ids:
            raise RuntimeError(f"Class '{cname}' not found in Open Images descriptions.")

    image_ids = collect_image_ids(labels_path, class_ids)

    selected_ids: dict[str, list[str]] = {}
    for cname in DEFAULT_CLASSES:
        available = image_ids.get(cname, [])
        if len(available) < args.count:
            raise RuntimeError(f"Insufficient images for '{cname}': requested {args.count}, available {len(available)}")
        rng.shuffle(available)
        selected_ids[cname] = available[: args.count]

    download_and_rename_images(
        downloader_script=downloader_path,
        selected_ids=selected_ids,
        classes=DEFAULT_CLASSES,
        dataset_dir=args.dest,
        metadata_dir=METADATA_DIR,
    )

    if HAS_RICH and console:
        summary_table = Table(title="Download Completed Successfully", border_style="green")
        summary_table.add_column("Class", style="cyan")
        summary_table.add_column("Saved Images", justify="right", style="bold white")
        summary_table.add_column("Folder Location", style="dim")

        for cname, folder_name in DEFAULT_CLASSES.items():
            folder = args.dest / folder_name
            count = len(list(folder.glob("*.jpg")))
            summary_table.add_row(cname, str(count), str(folder))
        console.print("\n", summary_table)
    else:
        print("\n" + "=" * 45)
        print(" DOWNLOAD COMPLETED")
        print("=" * 45)
        for cname, folder_name in DEFAULT_CLASSES.items():
            folder = args.dest / folder_name
            count = len(list(folder.glob("*.jpg")))
            print(f"  {cname:<10}: {count} images in {folder}")


if __name__ == "__main__":
    main()