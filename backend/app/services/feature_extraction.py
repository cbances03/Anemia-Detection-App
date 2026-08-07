from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.color import rgb2hsv, rgb2lab

from app.services.feature_validator import validate_feature_vector
from app.services.image_preprocessing import DEFAULT_IMAGE_SIZE, load_preprocessed_image


class FeatureExtractionError(ValueError):
    """Raised when color features cannot be extracted consistently."""


def extract_color_features(
    image_path: str | Path,
    img_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    use_slic: bool = False,
) -> np.ndarray:
    if use_slic is not False:
        raise FeatureExtractionError(
            "use_slic must be False for the deployed model."
        )

    img_rgb = load_preprocessed_image(image_path, img_size)

    try:
        arr = img_rgb.astype(np.float32) / 255.0
        lab = rgb2lab(arr)
        hsv = rgb2hsv(arr)

        features = []
        for space in [arr, lab, hsv]:
            for c in range(3):
                ch = space[:, :, c]
                features.extend(
                    [
                        np.mean(ch),
                        np.std(ch),
                        np.median(ch),
                        np.percentile(ch, 5),
                        np.percentile(ch, 10),
                        np.percentile(ch, 25),
                        np.percentile(ch, 75),
                        np.percentile(ch, 90),
                        np.percentile(ch, 95),
                        np.min(ch),
                        np.max(ch),
                    ]
                )

        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b_rgb = arr[:, :, 2]
        eps = 1e-6

        features.extend(
            [
                np.mean(r / (g + eps)),
                np.mean(r / (b_rgb + eps)),
                np.mean(g / (b_rgb + eps)),
                np.mean(r - g),
                np.mean(r - b_rgb),
                np.mean(g - b_rgb),
                np.std(r - g),
                np.std(r - b_rgb),
                np.std(g - b_rgb),
            ]
        )

        lightness = lab[:, :, 0]
        a = lab[:, :, 1]
        b_lab = lab[:, :, 2]

        features.extend(
            [
                np.mean(lightness),
                np.std(lightness),
                np.mean(a),
                np.std(a),
                np.mean(b_lab),
                np.std(b_lab),
                np.mean(a / (lightness + eps)),
                np.mean(b_lab / (lightness + eps)),
                np.mean((a + b_lab) / (lightness + eps)),
            ]
        )
    except (TypeError, ValueError, IndexError, FloatingPointError) as exc:
        raise FeatureExtractionError("Color feature extraction failed.") from exc

    return np.array(features, dtype=np.float32)


def extract_validated_features(
    image_path: str | Path,
    img_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    use_slic: bool = False,
) -> np.ndarray:
    features = extract_color_features(image_path, img_size, use_slic)
    return validate_feature_vector(features)
