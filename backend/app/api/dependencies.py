"""
Singleton services — initialized once, shared across requests.
This is how you'd do it in production (not recreating models per request).
"""
from functools import lru_cache
from backend.app.core.embedder import EmbeddingService
from backend.app.core.vector_store import VectorStore
from backend.app.core.llm_service import LLMService
from backend.app.core.rag_pipeline import RAGPipeline


@lru_cache()
def get_embedder() -> EmbeddingService:
    return EmbeddingService()

@lru_cache()
def get_vector_store() -> VectorStore:
    return VectorStore()

@lru_cache()
def get_llm_service() -> LLMService:
    return LLMService()

@lru_cache()
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
    )