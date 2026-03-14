from fastapi import APIRouter
from backend.app.models.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy", service="intelligent-doc-qa")