"""
Embedding Sercive to embed the text and queries using the Embedding model (from .env)
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from backend.app.config import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = None):
        model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # Embed a list of texts. Returns normalized embeddings.
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        # Embed a single query string.
        return self.embed_texts([query])[0]
