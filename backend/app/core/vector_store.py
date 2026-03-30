"""
ChromaDB wrapper. Decoupled from vendor — could swap to Pinecone/Weaviate later.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import chromadb
import numpy as np
from backend.app.config import settings
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_text: str
    metadata: Dict
    score: float  # 0-1, higher is better


class VectorStore:
    def __init__(self, persist_dir: str = None):
        persist_dir = persist_dir or settings.CHROMA_DIR
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info(f"ChromaDB initialized at: {persist_dir}")

    def _get_or_create_collection(self, collection_name: str):
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunks_texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict],
        ids: List[str],
        collection_name: str = "default"
    ):
        collection = self._get_or_create_collection(collection_name)
        collection.add(
            documents=chunks_texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Added {len(ids)} chunks to collection '{collection_name}'")

    def search(
        self,
        query_embedding: np.ndarray,
        collection_name: str = "default",
        top_k: int = 5,
    ) -> List[SearchResult]:
        collection = self._get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        search_results = []
        for i in range(len(results["ids"][0])):
            score = 1 - results["distances"][0][i]  # cosine distance → similarity
            search_results.append(SearchResult(
                chunk_text=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                score=round(score, 4),
            ))
        return search_results
    
    def get_all_documents(self, collection_name: str = "default") -> dict:
        """Retrieve all documents and metadata from a collection for BM25 indexing."""
        collection = self._get_or_create_collection(collection_name)
        results = collection.get(
            include=["documents", "metadatas"]
        )
        return results

    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete collection '{collection_name}': {e}")