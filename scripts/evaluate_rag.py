"""
Evaluation script for the RAG pipeline.
Measures retrieval quality and generation quality across different modes.
"""
import json
import time
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_eval_dataset(path: str) -> list:
	"""Load the evaluation dataset from JSON."""
	with open(path) as f:
		return json.load(f)
	
def evaluate_retrieval(question: str, expected_keywords: list, results: list) -> dict:
	"""
	Check if retrieved chunks contain the expected keywords.
	Returns recall (what fraction of expected keywords were found in retrieved chunks).
	"""
	# Combine all retrieved chunk texts into one string for searching
	retrieved_text = " ".join([r.chunk_text for r in results]).lower()

	# Check which keywords appear in retrieved chunks
	found = []
	missing = []
	for keyword in expected_keywords:
		if keyword.lower() in retrieved_text:
			found.append(keyword)
		else:
			missing.append(keyword)

	recall = len(found) / len(expected_keywords) if expected_keywords else 0.0

	return {
		"recall": round(recall, 4),
		"found_keywords": found,
		"missing_keywords": missing,
	}


def run_evaluation(mode: str = "hybrid", rerank: bool = True):
	"""
	Run all evaluation questions through the pipeline and collect metrics.
	mode: 'dense' or 'hybrid'
	rerank: whether to enable cross-encoder re-ranking
	"""
	from backend.app.core.embedder import EmbeddingService
	from backend.app.core.vector_store import VectorStore
	from backend.app.core.hybrid_retriever import HybridRetriever
	from backend.app.core.reranker import Reranker
	from backend.app.config import settings

	# Initialize services
	logger.info(f"Initializing services (mode={mode}, rerank={rerank})...")
	embedder = EmbeddingService()
	vector_store = VectorStore()
	hybrid_retriever = HybridRetriever(embedder, vector_store) if mode == "hybrid" else None
	reranker = Reranker() if rerank else None

	dataset = load_eval_dataset("backend/tests/fixtures/eval_dataset.json")
	logger.info(f"Running evaluation on {len(dataset)} questions...")

	results_log = []

	for i, item in enumerate(dataset):
		question = item["question"]
		expected_keywords = item["expected_keywords"]
		q_category = item["category"]

		# Retrieve chunks based on mode
		start_time = time.time()
		fetch_k = settings.TOP_K * 3 if rerank else settings.TOP_K

		if mode == "hybrid" and hybrid_retriever:
			retrieved = hybrid_retriever.search(
			    query=question,
			    collection_name="default",
			    top_k=fetch_k,
			    alpha=settings.HYBRID_ALPHA,
			)
		else:
			query_embedding = embedder.embed_query(question)
			retrieved = vector_store.search(
				query_embedding=query_embedding,
				collection_name="default",
				top_k=fetch_k,
			)

		# Re-rank if enabled
		if reranker and rerank:
			retrieved = reranker.rerank(question, retrieved, top_k=settings.TOP_K)

		retrieval_time = time.time() - start_time

		# Evaluate retrieval quality
		metrics = evaluate_retrieval(question, expected_keywords, retrieved)

		result = {
		    "question": question,
		    "category": q_category,
		    "recall": metrics["recall"],
		    "found_keywords": metrics["found_keywords"],
		    "missing_keywords": metrics["missing_keywords"],
		    "retrieval_time_ms": round(retrieval_time * 1000, 1),
		    "num_results": len(retrieved),
		}
		results_log.append(result)

		status = "PASS" if metrics["recall"] == 1.0 else "PARTIAL" if metrics["recall"] > 0 else "FAIL"
		print(f"  [{status}] Q{i+1} ({q_category}): recall={metrics['recall']:.2f} | {retrieval_time*1000:.0f}ms | {question[:50]}...")

	# Summary
	avg_recall = sum(r["recall"] for r in results_log) / len(results_log)
	perfect = sum(1 for r in results_log if r["recall"] == 1.0)
	avg_time = sum(r["retrieval_time_ms"] for r in results_log) / len(results_log)

	print(f"\n{'='*60}")
	print(f"MODE: {mode} | RERANK: {rerank}")
	print(f"Average Recall: {avg_recall:.4f}")
	print(f"Perfect Recall (1.0): {perfect}/{len(results_log)}")
	print(f"Average Retrieval Time: {avg_time:.1f}ms")
	print(f"{'='*60}")

	return results_log


if __name__ == "__main__":
	import sys

	mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
	rerank = sys.argv[2] != "false" if len(sys.argv) > 2 else True

	run_evaluation(mode=mode, rerank=rerank)