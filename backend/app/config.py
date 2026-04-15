"""
Centralized configuration via environment variables.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Intelligent Document Q&A"
    DEBUG: bool = True

    # Paths
    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "uploads")
    CHROMA_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "chroma_data")

    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int

    # Chunking
    CHUNK_SIZE: int = 512  # tokens
    CHUNK_OVERLAP: int = 50  # tokens

    # LLM — set ONE of these in your .env
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    LLM_PROVIDER: str
    LLM_MODEL: str

    # Retrieval
    TOP_K: int = 5
    HYBRID_ALPHA: float = 0.5
    RETRIEVAL_MODE: str = "hybrid"
    
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_ENABLED: bool = True


    class Config:
        env_file = "backend/.env"
        extra = "allow"


settings = Settings()
