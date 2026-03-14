from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from pathlib import Path
import shutil, uuid

from backend.app.api.dependencies import get_rag_pipeline
from backend.app.core.rag_pipeline import RAGPipeline
from backend.app.core.document_parser import ALLOWED_EXTENSIONS
from backend.app.models.schemas import IngestResponse
from backend.app.config import settings

router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Query(default="default"),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    # Save to disk
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4().hex}{ext}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check file size
    if file_path.stat().st_size > MAX_FILE_SIZE:
        file_path.unlink()
        raise HTTPException(413, "File too large. Max 50MB.")

    # Run ingestion pipeline
    try:
        result = pipeline.ingest_document(str(file_path), collection_name)
        return IngestResponse(**result)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")