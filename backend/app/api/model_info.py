from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.responses import ModelInfoResponse


router = APIRouter(tags=["Model"])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available",
        )

    model_metadata = bundle.metadata.get("model", {})
    model_status = model_metadata.get("status")
    decision_metadata = bundle.metadata.get("decision", {})
    metadata_threshold = decision_metadata.get("threshold")
    if not model_status or metadata_threshold is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not available",
        )

    return ModelInfoResponse(
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        threshold=metadata_threshold,
        status=model_status,
        positive_class=bundle.positive_class,
        positive_class_index=bundle.positive_class_index,
        feature_count=bundle.feature_count,
    )
