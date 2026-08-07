from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.model_loader import ModelLoadError, load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida el artefacto de producción.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        bundle = load_model_bundle(args.model, args.metadata)
    except ModelLoadError as exc:
        print("VALIDACIÓN FALLIDA")
        print(f"ERROR: {exc}")
        return 1

    print("VALIDACIÓN EXITOSA")
    print(f"PASS: modelo = {bundle.model_name} v{bundle.model_version}")
    print(f"PASS: threshold = {bundle.threshold:.2f}")
    print(f"PASS: entradas = {bundle.feature_count}")
    print(f"PASS: clase positiva = {bundle.positive_class}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
