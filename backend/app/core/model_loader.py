from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


EXPECTED_VERSION = "1.0.0"
EXPECTED_THRESHOLD = 0.40
EXPECTED_POSITIVE_CLASS = "Anemic"
EXPECTED_POSITIVE_CLASS_INDEX = 0
EXPECTED_FEATURES = 117
EXPECTED_K = 7
EXPECTED_METRIC = "manhattan"
EXPECTED_WEIGHTS = "distance"

REQUIRED_ARTIFACT_KEYS = {
    "modelo",
    "threshold",
    "anemic_index",
    "class_names",
    "img_size",
}

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "kNN_SMOTE_v1.0.0.joblib"
DEFAULT_METADATA_PATH = BACKEND_DIR / "models" / "metadata_v1.0.0.json"


class ModelLoadError(RuntimeError):
    """Raised when the production model cannot be loaded safely."""


@dataclass(frozen=True)
class ModelBundle:
    artifact: dict[str, Any]
    model: Any
    metadata: dict[str, Any]
    threshold: float
    positive_class: str
    positive_class_index: int
    model_version: str
    model_name: str
    feature_count: int


def _required(mapping: dict[str, Any], path: str, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ModelLoadError(f"Model metadata is invalid: missing {path}")
        current = current[key]
    return current


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"Model metadata is invalid: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ModelLoadError("Model metadata is invalid: expected a JSON object")
    return metadata


def _load_artifact(model_path: Path) -> dict[str, Any]:
    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Model artifact is invalid: {exc}") from exc

    if not isinstance(artifact, dict):
        raise ModelLoadError("Model artifact is invalid: expected a dictionary")

    missing = REQUIRED_ARTIFACT_KEYS.difference(artifact)
    if missing:
        raise ModelLoadError(
            f"Model artifact is invalid: missing keys {sorted(missing)}"
        )
    return artifact


def _find_classifier(model: Any) -> Any:
    if not hasattr(model, "named_steps"):
        raise ModelLoadError("Model validation failed: pipeline has no named_steps")

    steps = model.named_steps
    for name in ("clf", "knn", "classifier"):
        if name in steps:
            return steps[name]
    for step in steps.values():
        if step.__class__.__name__ == "KNeighborsClassifier":
            return step
    raise ModelLoadError("Model validation failed: KNeighborsClassifier not found")


def _feature_count(model: Any, classifier: Any) -> int:
    for name in ("scaler", "standardscaler"):
        step = model.named_steps.get(name)
        if step is not None and hasattr(step, "n_features_in_"):
            return int(step.n_features_in_)
    if hasattr(classifier, "n_features_in_"):
        return int(classifier.n_features_in_)
    raise ModelLoadError("Model validation failed: feature count unavailable")


def _validate_and_build(
    artifact: dict[str, Any], metadata: dict[str, Any]
) -> ModelBundle:
    model = artifact["modelo"]
    classifier = _find_classifier(model)
    feature_count = _feature_count(model, classifier)

    try:
        threshold = float(artifact["threshold"])
        classes = list(artifact["class_names"])
        positive_index = int(artifact["anemic_index"])
        model_name = str(_required(metadata, "model.name", "model", "name"))
        model_version = str(_required(metadata, "model.version", "model", "version"))
        metadata_threshold = float(
            _required(metadata, "decision.threshold", "decision", "threshold")
        )
        metadata_positive_class = str(
            _required(metadata, "dataset.positive_class", "dataset", "positive_class")
        )
        metadata_positive_index = int(
            _required(
                metadata,
                "dataset.positive_class_index",
                "dataset",
                "positive_class_index",
            )
        )
        metadata_features = int(
            _required(metadata, "input.feature_count", "input", "feature_count")
        )
    except (TypeError, ValueError) as exc:
        raise ModelLoadError(f"Model metadata is invalid: {exc}") from exc

    if positive_index < 0 or positive_index >= len(classes):
        raise ModelLoadError("Model validation failed: positive class index is invalid")
    positive_class = str(classes[positive_index])

    checks = (
        (math.isclose(threshold, EXPECTED_THRESHOLD, rel_tol=0, abs_tol=1e-9), "artifact threshold"),
        (math.isclose(metadata_threshold, EXPECTED_THRESHOLD, rel_tol=0, abs_tol=1e-9), "metadata threshold"),
        (model_version == EXPECTED_VERSION, "model version"),
        (positive_class == EXPECTED_POSITIVE_CLASS, "artifact positive class"),
        (metadata_positive_class == EXPECTED_POSITIVE_CLASS, "metadata positive class"),
        (positive_index == EXPECTED_POSITIVE_CLASS_INDEX, "artifact positive class index"),
        (metadata_positive_index == EXPECTED_POSITIVE_CLASS_INDEX, "metadata positive class index"),
        (int(classifier.n_neighbors) == EXPECTED_K, "n_neighbors"),
        (str(classifier.metric).lower() == EXPECTED_METRIC, "metric"),
        (str(classifier.weights).lower() == EXPECTED_WEIGHTS, "weights"),
        (feature_count == EXPECTED_FEATURES, "artifact feature count"),
        (metadata_features == EXPECTED_FEATURES, "metadata feature count"),
        (metadata_features == feature_count, "artifact/metadata feature count consistency"),
        (math.isclose(threshold, metadata_threshold, rel_tol=0, abs_tol=1e-9), "artifact/metadata threshold consistency"),
        (positive_index == metadata_positive_index, "artifact/metadata positive index consistency"),
    )
    for valid, field in checks:
        if not valid:
            raise ModelLoadError(f"Model validation failed: invalid {field}")

    return ModelBundle(
        artifact=artifact,
        model=model,
        metadata=metadata,
        threshold=threshold,
        positive_class=positive_class,
        positive_class_index=positive_index,
        model_version=model_version,
        model_name=model_name,
        feature_count=feature_count,
    )


def load_model_bundle(
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> ModelBundle:
    """Load and validate the production model exactly once at application startup."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.is_file():
        raise ModelLoadError(f"Model artifact not found: {model_path}")
    if not metadata_path.is_file():
        raise ModelLoadError(f"Model metadata not found: {metadata_path}")

    metadata = _load_metadata(metadata_path)
    artifact = _load_artifact(model_path)
    try:
        return _validate_and_build(artifact, metadata)
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Model validation failed: {exc}") from exc
