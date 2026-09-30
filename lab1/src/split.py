from collections import defaultdict
import math
from pathlib import Path
import random
from typing import NamedTuple, Sequence

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False

DEFAULT_SEED: int = 42
DEFAULT_TRAIN_RATIO: float = 0.70
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15

ImageSample = tuple[str, str]


class DatasetSplit(NamedTuple):
    train: list[ImageSample]
    validation: list[ImageSample]
    test: list[ImageSample]


def split_dataset(
    images: Sequence[tuple[str | Path, str]],
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    verbose: bool = True,
) -> DatasetSplit:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
        raise ValueError(f"Ratios must sum to 1.0 (got {total_ratio:.3f})")

    rng = random.Random(seed)

    class_groups: dict[str, list[str]] = defaultdict(list)
    for path, class_name in images:
        class_groups[class_name].append(str(path))

    train: list[ImageSample] = []
    validation: list[ImageSample] = []
    test: list[ImageSample] = []

    for class_name, paths in sorted(class_groups.items()):
        paths_copy = list(paths)
        rng.shuffle(paths_copy)

        total = len(paths_copy)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        train_paths = paths_copy[:train_count]
        val_paths = paths_copy[train_count: train_count + val_count]
        test_paths = paths_copy[train_count + val_count:]

        train.extend((p, class_name) for p in train_paths)
        validation.extend((p, class_name) for p in val_paths)
        test.extend((p, class_name) for p in test_paths)

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)

    if verbose:
        total_samples = len(train) + len(validation) + len(test)
        if HAS_RICH and console:
            table = Table(title=f"Dataset Split (Seed: {seed})", border_style="bright_blue")
            table.add_column("Subset", style="bold cyan")
            table.add_column("Sample Count", justify="right", style="green")
            table.add_column("Percentage", justify="right", style="yellow")

            table.add_row("Train", str(len(train)), f"{len(train)/total_samples:.1%}")
            table.add_row("Validation", str(len(validation)), f"{len(validation)/total_samples:.1%}")
            table.add_row("Test", str(len(test)), f"{len(test)/total_samples:.1%}")
            table.add_row("Total", str(total_samples), "100.0%", style="bold white")
            console.print(table)
        else:
            print("=" * 45)
            print(f" DATASET SPLIT (Seed: {seed})")
            print("=" * 45)
            print(f"Train:      {len(train):>6} ({len(train)/total_samples:.1%})")
            print(f"Validation: {len(validation):>6} ({len(validation)/total_samples:.1%})")
            print(f"Test:       {len(test):>6} ({len(test)/total_samples:.1%})")
            print("-" * 45)
            print(f"Total:      {total_samples:>6}")
            print("=" * 45)

    return DatasetSplit(train=train, validation=validation, test=test)