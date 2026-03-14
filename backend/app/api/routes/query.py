from fastapi import APIRouter, Depends
from backend.app.api.dependencies import get_rag_pipeline
from backend.app.core.rag_pipeline import RAGPipeline
from backend.app.models.schemas import QueryRequest, QueryResponseModel, SourceInfo

router = APIRouter()


@router.post("/query", response_model=QueryResponseModel)
async def query_documents(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    result = pipeline.query(request.question, request.collection_name)
    return QueryResponseModel(
        answer=result.answer,
        sources=[
            SourceInfo(text=s["text"][:300], source=s["source"], score=s["score"])
            for s in result.sources
        ],
        query=result.query,
    )