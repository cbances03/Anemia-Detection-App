from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib


REQUIRED_KEYS = {
    "modelo",
    "threshold",
    "anemic_index",
    "class_names",
    "img_size",
}

EXPECTED_THRESHOLD = 0.40
EXPECTED_K = 7
EXPECTED_METRIC = "manhattan"
EXPECTED_WEIGHTS = "distance"
EXPECTED_FEATURES = 117
EXPECTED_POSITIVE_CLASS = "Anemic"


class ValidationError(Exception):
    """Raised when the model artifact does not meet KAN-46 criteria."""


def load_artifact(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise ValidationError(f"No se encontró el artefacto: {model_path}")

    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        raise ValidationError(
            f"No se pudo deserializar el artefacto: {exc}"
        ) from exc

    if not isinstance(artifact, dict):
        raise ValidationError(
            "El artefacto debe ser un diccionario con las claves requeridas."
        )

    return artifact


def validate_required_keys(artifact: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS.difference(artifact.keys())
    if missing:
        raise ValidationError(
            f"Faltan claves requeridas: {sorted(missing)}"
        )


def validate_threshold(artifact: dict[str, Any]) -> None:
    threshold = float(artifact["threshold"])

    if not math.isclose(
        threshold,
        EXPECTED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValidationError(
            f"Threshold inválido: {threshold}. "
            f"Se esperaba {EXPECTED_THRESHOLD}."
        )


def get_pipeline_steps(artifact: dict[str, Any]) -> dict[str, Any]:
    model = artifact["modelo"]

    if not hasattr(model, "named_steps"):
        raise ValidationError(
            "El objeto 'modelo' no contiene un pipeline con named_steps."
        )

    return model.named_steps


def find_classifier(steps: dict[str, Any]) -> Any:
    for step_name in ("clf", "knn", "classifier"):
        if step_name in steps:
            return steps[step_name]

    for step in steps.values():
        if step.__class__.__name__ == "KNeighborsClassifier":
            return step

    raise ValidationError(
        "No se encontró un KNeighborsClassifier en el pipeline."
    )


def find_feature_count(
    steps: dict[str, Any],
    classifier: Any,
) -> int:
    for step_name in ("scaler", "standardscaler"):
        step = steps.get(step_name)
        if step is not None and hasattr(step, "n_features_in_"):
            return int(step.n_features_in_)

    if hasattr(classifier, "n_features_in_"):
        return int(classifier.n_features_in_)

    raise ValidationError(
        "No fue posible determinar el número de entradas del modelo."
    )


def validate_classifier(classifier: Any) -> None:
    if int(classifier.n_neighbors) != EXPECTED_K:
        raise ValidationError(
            f"k inválido: {classifier.n_neighbors}. "
            f"Se esperaba {EXPECTED_K}."
        )

    if str(classifier.metric).lower() != EXPECTED_METRIC:
        raise ValidationError(
            f"Métrica inválida: {classifier.metric}. "
            f"Se esperaba {EXPECTED_METRIC}."
        )

    if str(classifier.weights).lower() != EXPECTED_WEIGHTS:
        raise ValidationError(
            f"Pesos inválidos: {classifier.weights}. "
            f"Se esperaba {EXPECTED_WEIGHTS}."
        )


def validate_feature_count(feature_count: int) -> None:
    if feature_count != EXPECTED_FEATURES:
        raise ValidationError(
            f"Número de entradas inválido: {feature_count}. "
            f"Se esperaban {EXPECTED_FEATURES}."
        )


def validate_positive_class(artifact: dict[str, Any]) -> None:
    classes = list(artifact["class_names"])
    positive_index = int(artifact["anemic_index"])

    if positive_index < 0 or positive_index >= len(classes):
        raise ValidationError(
            f"Índice positivo fuera de rango: {positive_index}."
        )

    positive_class = str(classes[positive_index])

    if positive_class != EXPECTED_POSITIVE_CLASS:
        raise ValidationError(
            f"Clase positiva inválida: {positive_class}. "
            f"Se esperaba {EXPECTED_POSITIVE_CLASS}."
        )


def validate_metadata(
    metadata_path: Path | None,
    feature_count: int,
) -> None:
    if metadata_path is None:
        return

    if not metadata_path.exists():
        raise ValidationError(
            f"No se encontró el metadata: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if float(metadata["decision"]["threshold"]) != EXPECTED_THRESHOLD:
        raise ValidationError(
            "El threshold del metadata no coincide con 0.40."
        )

    if int(metadata["input"]["feature_count"]) != feature_count:
        raise ValidationError(
            "El número de features del metadata no coincide con el modelo."
        )

    if metadata["dataset"]["positive_class"] != EXPECTED_POSITIVE_CLASS:
        raise ValidationError(
            "La clase positiva del metadata no coincide con Anemic."
        )


def validate(
    model_path: Path,
    metadata_path: Path | None = None,
) -> list[str]:
    artifact = load_artifact(model_path)
    validate_required_keys(artifact)
    validate_threshold(artifact)

    steps = get_pipeline_steps(artifact)
    classifier = find_classifier(steps)
    validate_classifier(classifier)

    feature_count = find_feature_count(steps, classifier)
    validate_feature_count(feature_count)
    validate_positive_class(artifact)
    validate_metadata(metadata_path, feature_count)

    positive_index = int(artifact["anemic_index"])
    classes = list(artifact["class_names"])

    return [
        "PASS: el artefacto se deserializa correctamente",
        f"PASS: contiene las claves requeridas {sorted(REQUIRED_KEYS)}",
        f"PASS: threshold = {float(artifact['threshold']):.2f}",
        f"PASS: k = {int(classifier.n_neighbors)}",
        f"PASS: metric = {classifier.metric}",
        f"PASS: weights = {classifier.weights}",
        f"PASS: entradas = {feature_count}",
        f"PASS: clase positiva = {classes[positive_index]}",
        "PASS: metadata consistente con el artefacto",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida el artefacto del modelo según KAN-46."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Ruta al archivo .joblib versionado.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=False,
        help="Ruta opcional al metadata JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        results = validate(args.model, args.metadata)
    except ValidationError as exc:
        print("KAN-46: VALIDACIÓN FALLIDA")
        print(f"ERROR: {exc}")
        return 1

    print("KAN-46: VALIDACIÓN EXITOSA")
    for result in results:
        print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
