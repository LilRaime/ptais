from pathlib import Path
import cv2
import numpy as np

DEFAULT_IMAGE_SIZE: tuple[int, int] = (128, 128)


def preprocess_image(
    path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Failed to read image: {file_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0