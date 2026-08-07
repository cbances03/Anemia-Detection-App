import json
from pathlib import Path

import pytest

from app.core.model_loader import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    ModelLoadError,
    load_model_bundle,
)


def test_loads_valid_model_bundle():
    bundle = load_model_bundle()

    assert bundle.model_version == "1.0.0"
    assert bundle.threshold == pytest.approx(0.40)
    assert bundle.positive_class == "Anemic"
    assert bundle.positive_class_index == 0
    assert bundle.feature_count == 117


def test_fails_when_model_is_missing(tmp_path: Path):
    with pytest.raises(ModelLoadError, match="Model artifact not found"):
        load_model_bundle(tmp_path / "missing.joblib", DEFAULT_METADATA_PATH)


def test_fails_when_metadata_is_missing(tmp_path: Path):
    with pytest.raises(ModelLoadError, match="Model metadata not found"):
        load_model_bundle(DEFAULT_MODEL_PATH, tmp_path / "missing.json")


def test_fails_when_metadata_json_is_invalid(tmp_path: Path):
    metadata_path = tmp_path / "invalid.json"
    metadata_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ModelLoadError, match="Model metadata is invalid"):
        load_model_bundle(DEFAULT_MODEL_PATH, metadata_path)


def test_fails_when_required_metadata_is_missing(tmp_path: Path):
    metadata = json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))
    del metadata["model"]["version"]
    metadata_path = tmp_path / "missing-key.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelLoadError, match="missing model.version"):
        load_model_bundle(DEFAULT_MODEL_PATH, metadata_path)
