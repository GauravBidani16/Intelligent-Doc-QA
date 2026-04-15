"""
BM25 sparse retriever for keyword-based search.
Complements dense (vector) retrieval by catching exact keyword matches.
"""
from typing import List, Dict
from rank_bm25 import BM25Okapi
from backend.app.core.vector_store import VectorStore, SearchResult
import logging
import re

logger = logging.getLogger(__name__)

class BM25Retriever:
	def __init__(self):
		self.index = None
		self.documents = []
		self.metadatas = []
		self.ids = []

	def _tokenize(self, text: str) -> List[str]:
		#Simple word tokenization: lowercase, remove punctuation, split on whitespace
		text = text.lower()
		text = re.sub(r'[^\w\s]', ' ', text)
		tokens = text.split()
		return [t for t in tokens if len(t) > 1]
	
	def build_index(self, vector_store: VectorStore, collection_name: str = "default"):
		#Build BM25 index from all documents in a ChromaDB collection.
		results = vector_store.get_all_documents(collection_name)

		self.documents = results["documents"]
		self.metadatas = results["metadatas"]
		self.ids = results["ids"]

		# Tokenize all documents for BM25
		tokenized_docs = [self._tokenize(doc) for doc in self.documents]
		self.index = BM25Okapi(tokenized_docs)

		logger.info(f"BM25 index built with {len(self.documents)} documents")

	def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
		#Search the BM25 index and return ranked results.
		if self.index is None:
			logger.warning("BM25 index not built. Call build_index() first.")
			return []

		tokenized_query = self._tokenize(query)
		scores = self.index.get_scores(tokenized_query)

		# Get top_k indices sorted by score
		ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

		results = []
		for idx in ranked_indices:
			results.append(SearchResult(
			    chunk_text=self.documents[idx],
			    metadata=self.metadatas[idx],
			    score=float(scores[idx]),
			))

		return results