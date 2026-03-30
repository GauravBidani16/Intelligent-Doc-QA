"""
Cross-encoder re-ranker for precise relevance scoring.
"""
from typing import List
from sentence_transformers import CrossEncoder
from backend.app.core.vector_store import SearchResult
from backend.app.config import settings
import logging

logger = logging.getLogger(__name__)


class Reranker:
	def __init__(self, model_name: str = None):
		model_name = model_name or settings.RERANKER_MODEL
		logger.info(f"Loading re-ranker model: {model_name}")
		self.model = CrossEncoder(model_name)
		logger.info("Re-ranker model loaded.")

	def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
		"""Re-rank candidates using cross-encoder relevance scoring."""
		if not candidates:
			return candidates

		# Build query-chunk pairs for the cross-encoder
		pairs = [[query, c.chunk_text] for c in candidates]

		# Score all pairs in one batch
		scores = self.model.predict(pairs)

		# Attach new scores and sort
		reranked = []
		for i, candidate in enumerate(candidates):
			reranked.append(SearchResult(
				chunk_text=candidate.chunk_text,
				metadata=candidate.metadata,
				score=round(float(scores[i]), 4),
			))

		reranked.sort(key=lambda r: r.score, reverse=True)
		return reranked[:top_k]