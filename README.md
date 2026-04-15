# Intelligent Document Q&A System

A RAG (Retrieval-Augmented Generation) pipeline that allows users to upload documents and ask natural language questions, receiving accurate answers grounded in the document content with source citations. Built as a final project for CS 5100 - Fundamentals of Artificial Intelligence at Northeastern University, Spring 2026.

---

## Quick Look

Upload a document, then ask a question via the API:

Request:
```json
POST /api/query
{
  "question": "What is the pricing for Beacon RCA?"
}
```

Response:
```json
{
  "answer": "Beacon RCA is priced at $12 per host per month as an add-on for Professional tier customers. It is included at no additional cost for Enterprise tier customers. [Source 1, Source 2]",
  "sources": [
    { "source": "rag-test-kb.txt", "score": 0.91 },
    { "source": "rag-test-kb.txt", "score": 0.87 }
  ],
  "query": "What is the pricing for Beacon RCA?"
}
```

---

## Architecture

**Ingestion Pipeline**

File upload - Document parsing (PDF, DOCX, TXT) - Recursive chunking (512 tokens, 50 overlap) - Embedding (BAAI/bge-base-en-v1.5, 768d) - ChromaDB storage

**Query Pipeline**

Question embedding - Hybrid retrieval (BM25 + dense, 15 candidates) - Cross-encoder re-ranking (top 5) - LLM generation with source citation - Cited answer

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| API Framework | FastAPI | Backend server and endpoint routing |
| Embedding Model | BAAI/bge-base-en-v1.5 | Dense vector embeddings (768d) |
| Vector Store | ChromaDB | Persistent vector storage and similarity search |
| Sparse Retrieval | rank-bm25 | BM25 keyword-based retrieval |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder re-ranking for precision |
| LLM | Google Gemini | Answer generation |
| Document Parsing | PyMuPDF, python-docx | PDF and DOCX parsing |

---

## Project Structure

```
Inteligent-Doc-QA/
|- backend/
|   |- app/
|   |   |- main.py                  - FastAPI app entry point
|   |   |- config.py                - Settings and environment variable loading
|   |   |- api/
|   |   |   |- dependencies.py      - Singleton dependency injection
|   |   |   |- routes/
|   |   |       |- health.py        - GET /api/health
|   |   |       |- documents.py     - POST /api/documents/upload
|   |   |       |- query.py         - POST /api/query
|   |   |- core/
|   |   |   |- document_parser.py   - PDF, DOCX, TXT parsing
|   |   |   |- chunker.py           - Recursive text chunker
|   |   |   |- embedder.py          - Sentence-transformers wrapper
|   |   |   |- vector_store.py      - ChromaDB wrapper
|   |   |   |- bm25_retriever.py    - BM25 sparse retrieval
|   |   |   |- hybrid_retriever.py  - Dense and BM25 score fusion
|   |   |   |- reranker.py          - Cross-encoder re-ranking
|   |   |   |- llm_service.py       - Provider-agnostic LLM service
|   |   |   |- rag_pipeline.py      - Full pipeline orchestrator
|   |   |- models/
|   |       |- schemas.py           - Pydantic request and response models
|   |- tests/
|   |   |- fixtures/
|   |       |- eval_dataset.json    - 20-question evaluation dataset
|   |- requirements.txt
|   |- .env                         - Environment variables (gitignored)
|- scripts/
|   |- evaluate_rag.py              - Retrieval evaluation script
|   |- evaluate_generation.py       - Generation quality evaluation script
|   |- generation_eval_results.json - Generation evaluation results
|- docs/                            - Project documentation and screenshots
|- uploads/                         - Uploaded documents (gitignored)
|- chroma_data/                     - ChromaDB persistent storage (gitignored)
```

---

## Setup and Installation

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/Inteligent-Doc-QA.git
cd Inteligent-Doc-QA
```

**2. Create and activate a virtual environment**

```bash
python3.12 -m venv backend/venv
source backend/venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r backend/requirements.txt
```

**4. Configure environment variables**

Create `backend/.env` and set the following values:

```
GOOGLE_API_KEY=your-gemini-api-key
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.1-flash-lite-preview
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
RETRIEVAL_MODE=hybrid
HYBRID_ALPHA=0.5
RERANK_ENABLED=true
```

A free Gemini API key can be obtained at https://aistudio.google.com/apikey

---

## How to Run

**1. Start the server**

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

The server will be available at http://localhost:8000

**2. Open the Swagger UI**

Navigate to http://localhost:8000/docs in your browser. All endpoints are available here for interactive testing.

**3. Upload a document**

Use the POST /api/documents/upload endpoint to upload a PDF, DOCX, or TXT file. The response will confirm how many chunks were created.

If you do not have a document ready, a test knowledge base document and a list of 20 sample questions are available in backend/tests/fixtures/. Upload the knowledge base document and use the sample questions to verify the system is working correctly.

**4. Ask a question**

Use the POST /api/query endpoint with a JSON body:

```json
{
  "question": "Your question here"
}
```

The response will include the generated answer and the source chunks used.

---

## Evaluation

**Run retrieval evaluation**

```bash
python scripts/evaluate_rag.py hybrid true
```

Optional arguments: first argument is the retrieval mode (dense or hybrid), second is whether re-ranking is enabled (true or false).

**Run generation quality evaluation**

```bash
python scripts/evaluate_generation.py
```

Note: generation evaluation makes 40 Gemini API calls and takes approximately 9 minutes to complete at the free tier rate limit.

**Results - Embedding Model Comparison (Retrieval Recall)**

| Configuration | MiniLM-L6-v2 | BGE-base-en-v1.5 | Gain | % Improvement |
|---|---|---|---|---|
| Dense only | 0.8401 (11/20) | 0.9010 (13/20) | +0.061 | +7.3% |
| Dense + Rerank | 0.9383 (15/20) | 0.9383 (15/20) | +0.000 | 0.0% |
| Hybrid, no rerank | 0.8567 (13/20) | 0.9483 (16/20) | +0.092 | +10.7% |
| Hybrid + Rerank | 0.9383 (15/20) | 0.9383 (15/20) | +0.000 | 0.0% |

**Results - Retrieval Latency Comparison**

| Configuration | MiniLM Time | BGE Time | Change | % Change |
|---|---|---|---|---|
| Dense only | 18.8ms | 24.6ms | +5.8ms | +30.9% |
| Dense + Rerank | 134.2ms | 134.3ms | +0.1ms | +0.1% |
| Hybrid, no rerank | 20.0ms | 25.4ms | +5.4ms | +27.0% |
| Hybrid + Rerank | 133.8ms | 140.2ms | +6.4ms | +4.8% |

**Results - Generation Quality (hybrid + rerank, BGE embeddings)**

| Metric | Score |
|---|---|
| Average Faithfulness | 5.00 / 5.00 |
| Average Correctness | 4.75 / 5.00 |
| Questions scored | 20 / 20 |

---

## Known Limitations

- Multi-hop arithmetic: the system retrieves correct facts but does not perform calculations over them, resulting in incomplete answers on pricing or estimation questions that require combining multiple numbers.
- BM25 index rebuild: the BM25 index is rebuilt from scratch on every query rather than being cached, which adds overhead that would become significant at larger document scales.
- No frontend: the system is currently accessible only via the Swagger UI at /docs. A React-based chat interface is planned as a future improvement.

---

## Future Work

- Tool-augmented generation to allow the LLM to perform arithmetic over retrieved facts.
- BM25 index caching to reduce per-query latency at scale.
- Multimodal ingestion support for images, tables, and scanned documents.
- React chat interface with inline source citation display.

---

## Requirements

- Python 3.12
- Google Gemini API key (free tier available at https://aistudio.google.com/apikey)

---

## Author

Gaurav Bidani
MS Computer Science, Northeastern University
CS 5100 - Fundamentals of Artificial Intelligence, Spring 2026
