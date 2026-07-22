# Sistema de Detección No Invasiva de Anemia

## Descripción

Proyecto para la detección no invasiva de anemia mediante imágenes de la conjuntiva palpebral utilizando Inteligencia Artificial y una aplicación móvil.

---

## Estructura del proyecto

```text
anemia-detection/
├── frontend/      # Aplicación móvil
├── backend/       # API e integración con el modelo
├── ml/            # Desarrollo y experimentación del modelo
├── tests/         # Pruebas funcionales, integración y rendimiento
├── docs/          # Documentación del proyecto
├── evidence/      # Evidencias y reportes de validación
└── .github/       # Workflows de GitHub Actions
```

---

## Tecnologías

- React Native
- TypeScript
- Python
- FastAPI
- Scikit-Learn
- OpenCV
- Docker
- GitHub Actions

---

# Estrategia de ramas

El proyecto utiliza una estrategia basada en **Git Flow simplificado**, con el objetivo de mantener una versión estable del proyecto y facilitar el trabajo colaborativo.

## Ramas principales

### `main`

Contiene únicamente versiones estables del proyecto.

- No se realizan desarrollos directamente sobre esta rama.
- Toda integración debe realizarse mediante Pull Request.

### `develop`

Es la rama principal de desarrollo.

- Desde esta rama se crean las nuevas funcionalidades.
- Integra los cambios antes de pasar a `main`.

---

## Ramas de trabajo

### Nuevas funcionalidades

Formato:

```text
feature/<nombre>
```

Ejemplos:

```text
feature/HU-62
feature/HU-63
feature/mobile-camera
feature/backend-api
```

### Corrección de errores

Formato:

```text
fix/<nombre>
```

Ejemplos:

```text
fix/image-upload
fix/api-timeout
fix/camera-permissions
```

---

# Flujo de trabajo

Para cada Historia de Usuario se seguirá el siguiente proceso:

1. Actualizar la rama `develop`.
2. Crear una nueva rama `feature/*` o `fix/*`.
3. Implementar los cambios.
4. Realizar commits descriptivos.
5. Subir la rama al repositorio remoto.
6. Crear un Pull Request hacia `develop`.
7. Revisar y aprobar los cambios.
8. Fusionar la rama.
9. Eliminar la rama una vez integrada.

---

# Convención de commits

Se utiliza el estándar **Conventional Commits**.

Ejemplos:

```text
feat(HU-63): definir estrategia de ramas

feat(camera): implementar cámara guiada

fix(api): corregir timeout

docs(readme): actualizar documentación

test(prediction): agregar pruebas
```

---

# Pull Requests

Todo cambio debe integrarse mediante Pull Request.

Cada Pull Request debe incluir:

- Historia de Usuario relacionada.
- Descripción de los cambios.
- Evidencias (cuando aplique).
- Resultado de pruebas realizadas.
- Documentación actualizada (si corresponde).

No se permiten cambios directos sobre la rama `main`.

---

# Protección de ramas

La rama `main` cuenta con reglas de protección para garantizar la estabilidad del proyecto.

Se recomienda configurar:

- Pull Request obligatorio antes de fusionar cambios.
- Bloqueo de Force Push.
- Restricción de eliminación de la rama.
- Revisión de cambios antes de la integración.

---

# Instalación

Próximamente.

---

# Equipo

Proyecto académico desarrollado para la detección no invasiva de anemia mediante visión por computadora e Inteligencia Artificial.
