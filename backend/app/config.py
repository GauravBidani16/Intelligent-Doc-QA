"""
Centralized configuration via environment variables.
All secrets stay in .env, never hardcoded.
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

    # Embedding Model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Chunking
    CHUNK_SIZE: int = 512  # tokens
    CHUNK_OVERLAP: int = 50  # tokens

    # LLM — set ONE of these in your .env
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    LLM_PROVIDER: str = "gemini"  # "openai" or "anthropic"
    LLM_MODEL: str = "gemini-2.5-flash" # "gpt-4o-mini"  # or "claude-sonnet-4-20250514"

    # Retrieval
    TOP_K: int = 5

    class Config:
        env_file = "backend/.env"
        extra = "allow"


settings = Settings()
