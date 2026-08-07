from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.model_info import router as model_info_router
from app.api.predict import router as predict_router
from app.core.config import settings
from app.core.logging import logger
from app.core.model_loader import ModelLoadError, load_model_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_bundle = load_model_bundle()
    except ModelLoadError:
        logger.exception("Model loading failed during application startup")
        raise

    app.state.model_bundle = model_bundle
    logger.info(
        "Model loaded successfully: %s v%s | threshold=%.2f | features=%d",
        model_bundle.model_name,
        model_bundle.model_version,
        model_bundle.threshold,
        model_bundle.feature_count,
    )
    yield
    app.state.model_bundle = None

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(model_info_router)
app.include_router(predict_router)


@app.get("/")
def root():

    return JSONResponse(
        {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running"
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):

    logger.exception(exc)

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error"
        }
    )
