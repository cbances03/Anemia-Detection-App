from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.core.config import settings
from app.core.logging import logger

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION
)

app.include_router(health_router)
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
