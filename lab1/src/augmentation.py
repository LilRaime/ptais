import cv2
import numpy as np


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def rotation(image: np.ndarray, angle: float = 15.0) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT,
    )


def brightness_change(image: np.ndarray, factor: float = 1.3) -> np.ndarray:
    is_float = np.issubdtype(image.dtype, np.floating)
    max_val = 1.0 if is_float else 255.0
    scaled = image.astype(np.float32) * factor
    clipped = np.clip(scaled, 0.0, max_val)
    return clipped.astype(image.dtype)


def augment_image(
    image: np.ndarray,
    rotation_angle: float = 15.0,
    brightness_factor: float = 1.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flipped = horizontal_flip(image)
    rotated = rotation(image, angle=rotation_angle)
    brighter = brightness_change(image, factor=brightness_factor)
    return flipped, rotated, brighter