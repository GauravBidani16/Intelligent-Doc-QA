from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import health, documents, query
from backend.app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="RAG-based document Q&A system with cited answers",
    version="0.1.0",
)

# CORS — ready for when you add a frontend later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])
app.include_router(query.router, prefix="/api", tags=["Query"])


@app.on_event("startup")
async def startup():
    """Pre-load heavy models on startup so first request isn't slow."""
    from backend.app.api.dependencies import get_rag_pipeline
    get_rag_pipeline()  # Triggers model loading