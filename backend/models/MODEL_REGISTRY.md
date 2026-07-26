# Registro de modelos

## Convención de versionamiento

Se utilizará versionamiento semántico:

- **MAJOR**: cambia el algoritmo, la estructura de entrada o el significado de la salida.
- **MINOR**: se reentrena o mejora el modelo manteniendo la misma interfaz.
- **PATCH**: correcciones de metadatos, empaquetado o compatibilidad sin cambiar el comportamiento esperado.

Formato:

```text
NOMBRE_MODELO_vMAJOR.MINOR.PATCH.joblib
```

## Modelo vigente

| Campo | Valor |
|---|---|
| Nombre | kNN_SMOTE |
| Versión | 1.0.0 |
| Estado | production_candidate |
| Artefacto | `kNN_SMOTE_v1.0.0.joblib` |
| Metadatos | `metadata_v1.0.0.json` |
| Manifest | `model_manifest_v1.0.0.json` |
| SHA-256 | `381e1eb1f8b22877c9076a3420a3d14280ab52d46abdfaa4c847859a838e68b3` |
| Dataset | CP-ANEMIC |
| Tipo | Clasificación binaria |
| Clase positiva | Anemic |
| Threshold | 0.4 |
| Características | 117 |
| Tamaño de imagen | 224 × 224 |

## Historial

| Versión | Estado | Descripción |
|---|---|---|
| 1.0.0 | production_candidate | Primera versión formal del modelo ganador kNN + SMOTE. |

## Reglas

1. Nunca sobrescribir un artefacto versionado.
2. Cada nueva versión debe tener su propio metadata y manifest.
3. Toda versión debe conservar un hash SHA-256.
4. La aplicación debe registrar qué versión produjo cada predicción.
5. Una nueva versión solo reemplaza a la anterior después de pasar pruebas de validación.
6. Debe existir un mecanismo de rollback a la última versión estable.
