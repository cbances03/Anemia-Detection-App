from fastapi import APIRouter

from app.services.prediction_service import PredictionService

router = APIRouter(tags=["Prediction"])


@router.post("/predict")
def predict():

    return PredictionService.predict()
