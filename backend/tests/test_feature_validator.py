import inspect

import numpy as np
import pytest

from app.services import feature_validator
from app.services.feature_validator import (
    FeatureValidationError,
    validate_feature_vector,
)


def valid_features():
    return [float(index) for index in range(117)]


def test_accepts_exactly_117_numeric_values():
    result = validate_feature_vector(valid_features())

    np.testing.assert_array_equal(result[0], valid_features())


@pytest.mark.parametrize("size", [116, 118, 0])
def test_rejects_invalid_length_with_received_and_expected_counts(size):
    with pytest.raises(
        FeatureValidationError,
        match=rf"expected 117 values, received {size}",
    ):
        validate_feature_vector([0.0] * size)


def test_rejects_none_as_missing_value():
    features = valid_features()
    features[8] = None

    with pytest.raises(
        FeatureValidationError,
        match="missing values at positions: 8",
    ):
        validate_feature_vector(features)


def test_rejects_nan_as_missing_value():
    features = np.asarray(valid_features())
    features[12] = np.nan

    with pytest.raises(
        FeatureValidationError,
        match="missing values at positions: 12",
    ):
        validate_feature_vector(features)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_rejects_non_finite_values(value):
    features = valid_features()
    features[21] = value

    with pytest.raises(
        FeatureValidationError,
        match="non-finite values at positions: 21",
    ):
        validate_feature_vector(features)


def test_rejects_non_numeric_string():
    features = valid_features()
    features[34] = "not-a-number"

    with pytest.raises(
        FeatureValidationError,
        match="non-numeric values at positions: 34",
    ):
        validate_feature_vector(features)


def test_rejects_boolean_values():
    features = valid_features()
    features[3] = True

    with pytest.raises(
        FeatureValidationError,
        match="non-numeric values at positions: 3",
    ):
        validate_feature_vector(features)


def test_rejects_nested_array():
    with pytest.raises(FeatureValidationError, match="must be one-dimensional"):
        validate_feature_vector(np.zeros((117, 1)))


@pytest.mark.parametrize(
    "features",
    [tuple(range(117)), np.arange(117, dtype=np.int32)],
)
def test_accepts_tuple_and_numpy_numeric_types(features):
    result = validate_feature_vector(features)

    assert result.shape == (1, 117)
    assert result.dtype == np.float64


def test_output_has_inference_shape_and_numeric_dtype():
    result = validate_feature_vector(valid_features())

    assert result.shape == (1, 117)
    assert np.issubdtype(result.dtype, np.number)
    assert result.dtype == np.float64


def test_validator_does_not_transform_or_run_inference():
    source = inspect.getsource(feature_validator)
    forbidden_operations = (
        "StandardScaler",
        "SMOTE",
        ".predict(",
        ".predict_proba(",
    )

    for operation in forbidden_operations:
        assert operation not in source

    result = validate_feature_vector(valid_features())
    np.testing.assert_array_equal(result[0], valid_features())
