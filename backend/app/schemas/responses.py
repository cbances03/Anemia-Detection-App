from pydantic import BaseModel


class ServiceResponse(BaseModel):
    service: str
    version: str
    status: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    status: str
    positive_class: str
    positive_class_index: int
    feature_count: int
