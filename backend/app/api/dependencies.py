"""
Singleton services — initialized once, shared across requests.
"""
from functools import lru_cache
from backend.app.core.embedder import EmbeddingService
from backend.app.core.vector_store import VectorStore
from backend.app.core.llm_service import LLMService
from backend.app.core.hybrid_retriever import HybridRetriever
from backend.app.core.rag_pipeline import RAGPipeline
from backend.app.core.reranker import Reranker


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
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )

@lru_cache()
def get_reranker() -> Reranker:
    return Reranker()

@lru_cache()
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
        hybrid_retriever=get_hybrid_retriever(),
        reranker=get_reranker(),
    )