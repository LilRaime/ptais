"""Data augmentation functions for computer vision datasets."""

import cv2
import numpy as np


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Apply horizontal (left-to-right) flipping to an image.

    Args:
        image: Source image as a NumPy array (H, W, C) or (H, W).

    Returns:
        np.ndarray: Horizontally mirrored image.
    """
    return cv2.flip(image, 1)


def rotation(image: np.ndarray, angle: float = 15.0) -> np.ndarray:
    """Rotate the image around its center by a given angle in degrees.

    Uses BORDER_REFLECT to smoothly fill boundary pixels.

    Args:
        image: Source image as a NumPy array (H, W, C).
        angle: Rotation angle in degrees (positive = counter-clockwise).

    Returns:
        np.ndarray: Rotated image.
    """
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
    """Adjust image brightness by a scaling factor.

    Supports both uint8 [0..255] and float [0..1] image representations.

    Args:
        image: Source image array.
        factor: Brightness multiplier (>1.0 brightens, <1.0 darkens).

    Returns:
        np.ndarray: Brightness-adjusted image in original dtype.
    """
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
    """Generate three augmented variations: horizontal flip, rotation, and brightness change.

    Args:
        image: Original image array.
        rotation_angle: Angle for rotation augmentation.
        brightness_factor: Multiplier for brightness augmentation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (flipped, rotated, brighter)
    """
    flipped = horizontal_flip(image)
    rotated = rotation(image, angle=rotation_angle)
    brighter = brightness_change(image, factor=brightness_factor)
    return flipped, rotated, brighter