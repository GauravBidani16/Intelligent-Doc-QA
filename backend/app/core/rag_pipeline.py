"""
Ties everything together: parse -> chunk -> embed -> store -> retrieve -> generate.
"""
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import uuid
import logging

from backend.app.core.document_parser import get_parser, ExtractedDocument
from backend.app.core.chunker import RecursiveChunker, Chunk
from backend.app.core.embedder import EmbeddingService
from backend.app.core.vector_store import VectorStore, SearchResult
from backend.app.core.llm_service import LLMService
from backend.app.core.reranker import Reranker
from backend.app.core.hybrid_retriever import HybridRetriever
from backend.app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryResponse:
    answer: str
    sources: List[Dict]
    query: str


class RAGPipeline:
    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
        hybrid_retriever: HybridRetriever = None,
        reranker: Reranker = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.chunker = RecursiveChunker(
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP,
        )

    def ingest_document(self, file_path: str, collection_name: str = "default") -> Dict:
        """Full ingestion pipeline: parse -> chunk -> embed -> store."""
        # 1. Parse
        parser = get_parser(file_path)
        extracted = parser.parse(file_path)
        logger.info(f"Parsed '{file_path}': {len(extracted.text)} chars")

        # 2. Chunk
        chunks = self.chunker.chunk(
            extracted.text,
            doc_metadata={"source": Path(file_path).name}
        )
        logger.info(f"Created {len(chunks)} chunks")

        # 3. Embed
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)
        logger.info(f"Generated {embeddings.shape[0]} embeddings")

        # 4. Store in ChromaDB
        ids = [f"{Path(file_path).stem}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [
            {**c.metadata, "source": c.metadata.get("source", Path(file_path).name)}
            for c in chunks
        ]
        self.vector_store.add_chunks(
            chunks_texts=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
        )

        return {
            "filename": Path(file_path).name,
            "chunks": len(chunks),
            "status": "ready",
        }

    def query(self, question: str, collection_name: str = "default") -> QueryResponse:
        """Full RAG query: retrieve -> re-rank -> generate answer."""
        # 1. Retrieve candidates (hybrid or dense)
        fetch_k = settings.TOP_K * 3 if self.reranker and settings.RERANK_ENABLED else settings.TOP_K

        if settings.RETRIEVAL_MODE == "hybrid" and self.hybrid_retriever:
            results = self.hybrid_retriever.search(
                query=question,
                collection_name=collection_name,
                top_k=fetch_k,
                alpha=settings.HYBRID_ALPHA,
            )
        else:
            query_embedding = self.embedder.embed_query(question)
            results = self.vector_store.search(
                query_embedding=query_embedding,
                collection_name=collection_name,
                top_k=fetch_k,
            )

        if not results:
            return QueryResponse(
                answer="No relevant documents found. Please upload documents first.",
                sources=[],
                query=question,
            )

        # 2. Re-rank if enabled
        if self.reranker and settings.RERANK_ENABLED:
            results = self.reranker.rerank(question, results, top_k=settings.TOP_K)

        # 3. Build context for LLM
        context_chunks = [
            {"text": r.chunk_text, "source": r.metadata.get("source", "unknown"), "score": r.score}
            for r in results
        ]

        # 4. Generate answer
        answer = self.llm_service.generate(question, context_chunks)

        return QueryResponse(
            answer=answer,
            sources=context_chunks,
            query=question,
        )