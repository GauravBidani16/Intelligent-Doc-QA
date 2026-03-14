from pydantic import BaseModel
from typing import List, Dict, Optional


class HealthResponse(BaseModel):
    status: str
    service: str


class IngestResponse(BaseModel):
    filename: str
    chunks: int
    status: str


class QueryRequest(BaseModel):
    question: str
    collection_name: str = "default"


class SourceInfo(BaseModel):
    text: str
    source: str
    score: float


class QueryResponseModel(BaseModel):
    answer: str
    sources: List[SourceInfo]
    query: str