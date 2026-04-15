"""
Hybrid retriever: combines dense (vector) and sparse (BM25) retrieval
with score normalization and weighted fusion.
"""
from typing import List, Dict
from backend.app.core.vector_store import VectorStore, SearchResult
from backend.app.core.bm25_retriever import BM25Retriever
from backend.app.core.embedder import EmbeddingService
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
	def __init__(self, embedder: EmbeddingService, vector_store: VectorStore):
		self.embedder = embedder
		self.vector_store = vector_store
		self.bm25 = BM25Retriever()

	def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
		"""Min-max normalize scores to 0-1 range."""
		if not results:
			return results

		scores = [r.score for r in results]
		min_score = min(scores)
		max_score = max(scores)

		# All scores identical - give everything equal weight
		if max_score == min_score:
			return [
				SearchResult(chunk_text=r.chunk_text, metadata=r.metadata, score=1.0)
				for r in results
			]

		return [
			SearchResult(
				chunk_text=r.chunk_text,
				metadata=r.metadata,
				score=(r.score - min_score) / (max_score - min_score),
			)
			for r in results
		]
	
	def search(self, query: str, collection_name: str = "default", top_k: int = 5, alpha: float = 0.7,) -> List[SearchResult]:
		"""
		Hybrid search combining dense and BM25 retrieval.
		alpha = 1 is pure dense, alpha = 0 is pure BM25.
		"""
		# Build BM25 index from current collection
		self.bm25.build_index(self.vector_store, collection_name)

		# Run both searches (fetch more than top_k so fusion has enough to work with)
		fetch_k = top_k * 3

		# Dense search
		query_embedding = self.embedder.embed_query(query)
		dense_results = self.vector_store.search(query_embedding, collection_name, fetch_k)

		# BM25 search
		bm25_results = self.bm25.search(query, fetch_k)

		# Normalize both score sets to 0-1
		dense_normalized = self._normalize_scores(dense_results)
		bm25_normalized = self._normalize_scores(bm25_results)

		# Build lookup dicts keyed by chunk text
		dense_map = {r.chunk_text: r.score for r in dense_normalized}
		bm25_map = {r.chunk_text: r.score for r in bm25_normalized}

		# Get all unique chunks from both result sets
		all_chunks = set(dense_map.keys()) | set(bm25_map.keys())

		# Compute fused scores
		fused_results = []
		for chunk_text in all_chunks:
			dense_score = dense_map.get(chunk_text, 0.0)
			bm25_score = bm25_map.get(chunk_text, 0.0)
			fused_score = alpha * dense_score + (1 - alpha) * bm25_score

			# Find metadata from whichever search returned this chunk
			metadata = {}
			for r in dense_results + bm25_results:
				if r.chunk_text == chunk_text:
					metadata = r.metadata
					break

			fused_results.append(SearchResult(
				chunk_text=chunk_text,
				metadata=metadata,
				score=round(fused_score, 4),
			))

		# Sort by fused score and return top_k
		fused_results.sort(key=lambda r: r.score, reverse=True)
		return fused_results[:top_k]
