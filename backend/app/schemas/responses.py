from pydantic import BaseModel


class ServiceResponse(BaseModel):
    service: str
    version: str
    status: str


class HealthResponse(BaseModel):
    status: str
