from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


EXPECTED_FEATURE_COUNT = 117


class FeatureValidationError(ValueError):
    """Raised when a feature vector is not safe to send to the model."""


def _format_positions(positions: list[int]) -> str:
    return ", ".join(str(position) for position in positions)


def validate_feature_vector(
    features: Sequence[Any] | np.ndarray,
    expected_count: int = EXPECTED_FEATURE_COUNT,
) -> np.ndarray:
    """Validate one feature vector and return a float64 row for inference."""
    if isinstance(features, (str, bytes)) or not isinstance(
        features, (Sequence, np.ndarray)
    ):
        raise FeatureValidationError(
            "Feature vector must be a list, tuple, or one-dimensional ndarray."
        )

    if isinstance(features, np.ndarray) and features.ndim != 1:
        raise FeatureValidationError(
            "Feature vector must be one-dimensional; "
            f"received array shape {features.shape}."
        )

    try:
        received_count = len(features)
    except TypeError as exc:
        raise FeatureValidationError(
            "Feature vector must have a measurable length."
        ) from exc

    if received_count != expected_count:
        raise FeatureValidationError(
            "Invalid feature vector length: "
            f"expected {expected_count} values, received {received_count}."
        )

    missing_positions: list[int] = []
    non_numeric_positions: list[int] = []
    converted_values: list[float] = []

    for index, value in enumerate(features):
        if value is None:
            missing_positions.append(index)
            converted_values.append(math.nan)
            continue

        if isinstance(value, (bool, np.bool_)):
            non_numeric_positions.append(index)
            converted_values.append(math.nan)
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            non_numeric_positions.append(index)
            converted_values.append(math.nan)
            continue

        if math.isnan(numeric_value):
            missing_positions.append(index)
        converted_values.append(numeric_value)

    if missing_positions:
        raise FeatureValidationError(
            "Feature vector contains missing values at positions: "
            f"{_format_positions(missing_positions)}."
        )

    if non_numeric_positions:
        raise FeatureValidationError(
            "Feature vector contains non-numeric values at positions: "
            f"{_format_positions(non_numeric_positions)}."
        )

    validated = np.asarray(converted_values, dtype=np.float64)
    non_finite_positions = np.flatnonzero(~np.isfinite(validated)).tolist()
    if non_finite_positions:
        raise FeatureValidationError(
            "Feature vector contains non-finite values at positions: "
            f"{_format_positions(non_finite_positions)}."
        )

    return validated.reshape(1, expected_count)
