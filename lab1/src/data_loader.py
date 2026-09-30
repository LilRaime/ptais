from pathlib import Path
from typing import NamedTuple, Sequence
import numpy as np

from .preprocessing import DEFAULT_IMAGE_SIZE, preprocess_image

ImageSample = tuple[str | Path, str]


class LoadedDataset(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    class_names: list[str]


def load_dataset(
    samples: Sequence[ImageSample],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    verbose: bool = False,
) -> LoadedDataset:
    class_names = sorted({class_name for _, class_name in samples})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for path, class_name in samples:
        try:
            image = preprocess_image(path, image_size=image_size)
            X_list.append(image)
            y_list.append(class_to_idx[class_name])
        except Exception as err:
            if verbose:
                print(f"[!] Warning: failed to load {path}: {err}")

    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, *image_size[::-1], 3), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64) if y_list else np.empty((0,), dtype=np.int64)

    return LoadedDataset(X=X, y=y, class_names=class_names)