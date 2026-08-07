import inspect
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services import feature_extraction, image_preprocessing
from app.services.feature_extraction import (
    FeatureExtractionError,
    extract_color_features,
    extract_validated_features,
)
from app.services.feature_validator import FeatureValidationError, validate_feature_vector
from app.services.image_preprocessing import (
    ImagePreprocessingError,
    clahe_lab_correction,
    gray_world_white_balance,
    load_preprocessed_image,
)


@pytest.fixture
def synthetic_image_path(tmp_path: Path) -> Path:
    y, x = np.indices((48, 64))
    image = np.stack(
        ((x * 4) % 256, (y * 5) % 256, ((x + y) * 3) % 256),
        axis=2,
    ).astype(np.uint8)
    path = tmp_path / "synthetic.png"
    Image.fromarray(image, mode="RGB").save(path)
    return path


def test_gray_world_preserves_shape_and_returns_uint8():
    image = np.full((20, 30, 3), [180, 90, 45], dtype=np.uint8)

    result = gray_world_white_balance(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_clahe_preserves_shape_and_dtype():
    image = np.full((20, 30, 3), [180, 90, 45], dtype=np.uint8)

    result = clahe_lab_correction(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_preprocessing_returns_expected_shape(synthetic_image_path):
    result = load_preprocessed_image(synthetic_image_path)

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_extraction_returns_117_finite_numeric_features(synthetic_image_path):
    features = extract_color_features(synthetic_image_path, use_slic=False)

    assert features.shape == (117,)
    assert features.dtype == np.float32
    assert np.issubdtype(features.dtype, np.number)
    assert np.isfinite(features).all()


def test_kan32_accepts_extracted_features(synthetic_image_path):
    features = extract_color_features(synthetic_image_path)

    validated = validate_feature_vector(features)

    assert validated.shape == (1, 117)
    assert validated.dtype == np.float64


def test_integrated_output_has_final_inference_shape(synthetic_image_path):
    validated = extract_validated_features(synthetic_image_path, use_slic=False)

    assert validated.shape == (1, 117)
    assert validated.dtype == np.float64
    assert np.isfinite(validated).all()


def test_use_slic_true_is_rejected(synthetic_image_path):
    with pytest.raises(FeatureExtractionError, match="use_slic must be False"):
        extract_color_features(synthetic_image_path, use_slic=True)


def test_missing_image_has_controlled_error(tmp_path):
    missing_path = tmp_path / "missing.jpg"

    with pytest.raises(ImagePreprocessingError, match="Image file not found"):
        extract_validated_features(missing_path)


def test_corrupt_image_has_controlled_error(tmp_path):
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"not an image")

    with pytest.raises(
        ImagePreprocessingError,
        match="Image file is invalid or could not be read",
    ):
        extract_validated_features(corrupt_path)


def test_integrated_flow_rejects_wrong_feature_count(monkeypatch):
    monkeypatch.setattr(
        feature_extraction,
        "extract_color_features",
        lambda *_args, **_kwargs: np.zeros(116, dtype=np.float32),
    )

    with pytest.raises(FeatureValidationError, match="expected 117.*received 116"):
        extract_validated_features("unused.jpg")


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_integrated_flow_rejects_missing_or_non_finite_features(
    monkeypatch,
    invalid_value,
):
    features = np.zeros(117, dtype=np.float32)
    features[42] = invalid_value
    monkeypatch.setattr(
        feature_extraction,
        "extract_color_features",
        lambda *_args, **_kwargs: features,
    )

    with pytest.raises(FeatureValidationError, match="positions: 42"):
        extract_validated_features("unused.jpg")


def test_production_modules_do_not_infer_scale_resample_or_reload_model():
    source = inspect.getsource(image_preprocessing) + inspect.getsource(
        feature_extraction
    )

    for forbidden_operation in (
        ".predict(",
        ".predict_proba(",
        "StandardScaler",
        "SMOTE",
        "joblib.load",
    ):
        assert forbidden_operation not in source
