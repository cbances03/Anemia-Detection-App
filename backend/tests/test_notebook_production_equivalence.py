from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab

from app.services.feature_extraction import extract_color_features


CONTROL_IMAGE_PATH = Path(__file__).parent / "fixtures" / "cp_anemic_control.jpg"


def notebook_reference_features(image_path, img_size=(224, 224), use_slic=False):
    # Independent copy of the notebook source; intentionally does not import
    # production preprocessing or extraction functions.
    def gray_world_white_balance(img_rgb):
        img = img_rgb.astype(np.float32)
        avg_r = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_b = np.mean(img[:, :, 2])
        avg_gray = (avg_r + avg_g + avg_b) / 3
        img[:, :, 0] *= avg_gray / (avg_r + 1e-6)
        img[:, :, 1] *= avg_gray / (avg_g + 1e-6)
        img[:, :, 2] *= avg_gray / (avg_b + 1e-6)
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def clahe_lab_correction(img_rgb):
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    img = Image.open(image_path).convert("RGB").resize(img_size)
    img_rgb = clahe_lab_correction(
        gray_world_white_balance(np.array(img).astype(np.uint8))
    )
    arr = img_rgb.astype(np.float32) / 255.0
    lab = rgb2lab(arr)
    hsv = rgb2hsv(arr)
    features = []

    for space in [arr, lab, hsv]:
        for c in range(3):
            ch = space[:, :, c]
            features.extend(
                [
                    np.mean(ch), np.std(ch), np.median(ch),
                    np.percentile(ch, 5), np.percentile(ch, 10),
                    np.percentile(ch, 25), np.percentile(ch, 75),
                    np.percentile(ch, 90), np.percentile(ch, 95),
                    np.min(ch), np.max(ch),
                ]
            )

    r, g, b_rgb = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    eps = 1e-6
    features.extend(
        [
            np.mean(r / (g + eps)), np.mean(r / (b_rgb + eps)),
            np.mean(g / (b_rgb + eps)), np.mean(r - g),
            np.mean(r - b_rgb), np.mean(g - b_rgb), np.std(r - g),
            np.std(r - b_rgb), np.std(g - b_rgb),
        ]
    )
    lightness, a, b_lab = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    features.extend(
        [
            np.mean(lightness), np.std(lightness), np.mean(a), np.std(a),
            np.mean(b_lab), np.std(b_lab),
            np.mean(a / (lightness + eps)),
            np.mean(b_lab / (lightness + eps)),
            np.mean((a + b_lab) / (lightness + eps)),
        ]
    )
    return np.array(features, dtype=np.float32)


@pytest.mark.skipif(
    not CONTROL_IMAGE_PATH.is_file(),
    reason=f"Place a real CP-ANEMIC image at {CONTROL_IMAGE_PATH}",
)
def test_real_control_image_matches_notebook_reference():
    notebook_features = notebook_reference_features(
        CONTROL_IMAGE_PATH,
        use_slic=False,
    )
    production_features = extract_color_features(
        CONTROL_IMAGE_PATH,
        use_slic=False,
    )

    assert notebook_features.shape == (117,)
    assert production_features.shape == (117,)
    np.testing.assert_allclose(
        notebook_features,
        production_features,
        rtol=1e-6,
        atol=1e-6,
    )
