from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


DEFAULT_IMAGE_SIZE = (224, 224)
GRAY_WORLD_EPSILON = 1e-6
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot be read or preprocessed safely."""


def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32)

    avg_r = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_b = np.mean(img[:, :, 2])
    avg_gray = (avg_r + avg_g + avg_b) / 3

    img[:, :, 0] *= avg_gray / (avg_r + GRAY_WORLD_EPSILON)
    img[:, :, 1] *= avg_gray / (avg_g + GRAY_WORLD_EPSILON)
    img[:, :, 2] *= avg_gray / (avg_b + GRAY_WORLD_EPSILON)

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def clahe_lab_correction(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    corrected = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return corrected


def load_preprocessed_image(
    image_path: str | Path,
    img_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    path = Path(image_path)
    if not path.is_file():
        raise ImagePreprocessingError(f"Image file not found: {path}")

    try:
        with Image.open(path) as source_image:
            img = source_image.convert("RGB").resize(img_size)
            arr = np.array(img).astype(np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImagePreprocessingError(
            f"Image file is invalid or could not be read: {path}"
        ) from exc

    try:
        arr = gray_world_white_balance(arr)
        arr = clahe_lab_correction(arr)
    except (cv2.error, TypeError, ValueError, IndexError) as exc:
        raise ImagePreprocessingError(
            f"Image preprocessing failed: {path}"
        ) from exc

    return arr
