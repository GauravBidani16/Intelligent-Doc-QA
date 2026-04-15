"""
Generation quality evaluation script.

Runs all 20 questions through the full RAG pipeline (hybrid + rerank),
then scores each generated answer for faithfulness and correctness
using LLM as a judge (Currently Gemini 3.1 Flash).

Note: This script takes few minutes to run depending on the number of questions
that are being evaluated (I evaluated 20 Questions ~8 mins)
"""
import json
import time
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import google.generativeai as genai
from backend.app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_DELAY_SECONDS = 13


def load_eval_dataset(path: str) -> list:
    """Load the evaluation dataset from JSON."""
    with open(path) as f:
        return json.load(f)


def build_judge_prompt(question, context, generated_answer, expected_answer):
    
    return f"""You are evaluating a RAG (Retrieval-Augmented Generation) system's answer quality.

Question: {question}

Retrieved Context (what the system had access to):
{context}

Generated Answer (what the system produced):
{generated_answer}

Expected Answer (the correct answer):
{expected_answer}

Score the generated answer on TWO dimensions, each from 1 to 5:

FAITHFULNESS (1-5): Does the generated answer only use information from the retrieved context?
- 5: Everything stated is directly supported by the context
- 4: Almost all claims are in the context, very minor additions
- 3: Most claims are in the context, some unsupported additions
- 2: Several claims go beyond what the context says
- 1: Answer contains significant hallucinations not in context

CORRECTNESS (1-5): How well does the generated answer match the expected answer?
- 5: Complete and accurate, covers all key points
- 4: Mostly correct, minor details missing or slightly off
- 3: Partially correct, some key information missing
- 2: Significant parts wrong or missing
- 1: Mostly incorrect or irrelevant

Respond ONLY with valid JSON in this exact format, no extra text, no markdown:
{{
  "faithfulness_score": <number 1-5>,
  "faithfulness_reason": "<one sentence explanation>",
  "correctness_score": <number 1-5>,
  "correctness_reason": "<one sentence explanation>"
}}"""


def score_answer(judge_model, question, context_chunks, generated_answer, expected_answer):
    """
    Send the generated answer to Gemini for scoring.
    Returns a dict with faithfulness and correctness scores.
    """
    # Build a readable context string from retrieved chunks
    context = "\n\n".join([
        f"[Chunk {i+1}]: {chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])

    prompt = build_judge_prompt(question, context, generated_answer, expected_answer)

    response = judge_model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 1024,
        }
    )

    # Print raw response so we can see exactly what Gemini returned
    raw = response.text.strip()

    # Clean markdown fences if present
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def run_generation_evaluation(limit=None):
    """
    Main function - runs all 20 questions through generation + scoring.
    """
    from backend.app.core.embedder import EmbeddingService
    from backend.app.core.vector_store import VectorStore
    from backend.app.core.hybrid_retriever import HybridRetriever
    from backend.app.core.reranker import Reranker
    from backend.app.core.llm_service import LLMService

    # Initialize all pipeline services
    logger.info("Initializing services...")
    embedder = EmbeddingService()
    vector_store = VectorStore()
    hybrid_retriever = HybridRetriever(embedder, vector_store)
    reranker = Reranker()
    llm_service = LLMService()

    # Initialize a separate Gemini model instance for judging
    # This uses the same API key and model as the main LLM service
    genai.configure(api_key=settings.GOOGLE_API_KEY)
    judge_model = genai.GenerativeModel(settings.LLM_MODEL)

    # Load the 20-question evaluation dataset
    dataset = load_eval_dataset("backend/tests/fixtures/eval_dataset.json")
    total = len(dataset)
    total_calls = total * 2
    estimated_minutes = (total_calls * GEMINI_DELAY_SECONDS) // 60 + 1

    logger.info(f"Loaded {total} questions")
    logger.info(f"Total Gemini calls: {total_calls}")
    logger.info(f"Estimated runtime: ~{estimated_minutes} minutes")

    results = []
    fetch_k = settings.TOP_K * 3  # Fetch more candidates, reranker narrows to TOP_K

    if limit:
        dataset = dataset[:limit]

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_answer = item["expected_answer"]
        category = item["category"]
        difficulty = item["difficulty"]

        print(f"\n[{i+1}/{total}] {question[:65]}...")

        # --- Phase 1: Retrieve chunks ---
        retrieved = hybrid_retriever.search(
            query=question,
            collection_name="default",
            top_k=fetch_k,
            alpha=settings.HYBRID_ALPHA,
        )
        retrieved = reranker.rerank(question, retrieved, top_k=settings.TOP_K)

        # Build context_chunks in the format LLMService expects
        context_chunks = [
            {
                "text": r.chunk_text,
                "source": r.metadata.get("source", "unknown"),
                "score": r.score,
            }
            for r in retrieved
        ]

        # --- Phase 2: Generate answer (Gemini call #1) ---
        print(f"  Generating answer (call {i*2 + 1}/{total_calls})...")
        generated_answer = llm_service.generate(question, context_chunks)
        print(f"  Done. Waiting {GEMINI_DELAY_SECONDS}s for rate limit...")
        time.sleep(GEMINI_DELAY_SECONDS)

        # --- Phase 3: Score the answer (Gemini call #2) ---
        print(f"  Scoring answer (call {i*2 + 2}/{total_calls})...")
        try:
            scores = score_answer(
                judge_model,
                question,
                context_chunks,
                generated_answer,
                expected_answer,
            )
            faithfulness = scores["faithfulness_score"]
            correctness = scores["correctness_score"]
            faith_reason = scores["faithfulness_reason"]
            correct_reason = scores["correctness_reason"]
        except Exception as e:
            # If scoring fails (JSON parse error, API error), log and continue
            logger.warning(f"  Scoring failed for Q{i+1}: {e}")
            faithfulness = 0
            correctness = 0
            faith_reason = f"Scoring error: {str(e)}"
            correct_reason = f"Scoring error: {str(e)}"

        print(f"  Faithfulness: {faithfulness}/5 | Correctness: {correctness}/5")
        print(f"  Waiting {GEMINI_DELAY_SECONDS}s for rate limit...")
        time.sleep(GEMINI_DELAY_SECONDS)

        results.append({
            "id": item.get("id", i + 1),
            "question": question,
            "category": category,
            "difficulty": difficulty,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "faithfulness_score": faithfulness,
            "faithfulness_reason": faith_reason,
            "correctness_score": correctness,
            "correctness_reason": correct_reason,
        })

    # --- Aggregate results ---
    # Exclude any questions where scoring failed (score = 0)
    valid = [r for r in results if r["faithfulness_score"] > 0]

    avg_faith = sum(r["faithfulness_score"] for r in valid) / len(valid)
    avg_correct = sum(r["correctness_score"] for r in valid) / len(valid)

    print(f"\n{'='*60}")
    print(f"GENERATION QUALITY RESULTS")
    print(f"Config: hybrid + rerank")
    print(f"Questions scored: {len(valid)}/{total}")
    print(f"Average Faithfulness: {avg_faith:.2f} / 5.00")
    print(f"Average Correctness:  {avg_correct:.2f} / 5.00")
    print(f"{'='*60}")

    print("\nBy Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in valid if r["difficulty"] == diff]
        if subset:
            f = sum(r["faithfulness_score"] for r in subset) / len(subset)
            c = sum(r["correctness_score"] for r in subset) / len(subset)
            print(f"  {diff.capitalize()} ({len(subset)} questions): "
                  f"Faithfulness={f:.2f}  Correctness={c:.2f}")

    print("\nBy Category:")
    categories = sorted(set(r["category"] for r in valid))
    for cat in categories:
        subset = [r for r in valid if r["category"] == cat]
        f = sum(r["faithfulness_score"] for r in subset) / len(subset)
        c = sum(r["correctness_score"] for r in subset) / len(subset)
        print(f"  {cat} ({len(subset)}): Faithfulness={f:.2f}  Correctness={c:.2f}")

    # Save full results to JSON
    output = {
        "summary": {
            "total_questions": total,
            "questions_scored": len(valid),
            "avg_faithfulness": round(avg_faith, 4),
            "avg_correctness": round(avg_correct, 4),
            "config": "hybrid + rerank",
        },
        "by_difficulty": {},
        "by_category": {},
        "results": results,
    }

    # Add difficulty breakdown to summary
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in valid if r["difficulty"] == diff]
        if subset:
            output["by_difficulty"][diff] = {
                "count": len(subset),
                "avg_faithfulness": round(sum(r["faithfulness_score"] for r in subset) / len(subset), 4),
                "avg_correctness": round(sum(r["correctness_score"] for r in subset) / len(subset), 4),
            }

    # Add category breakdown to summary
    for cat in categories:
        subset = [r for r in valid if r["category"] == cat]
        output["by_category"][cat] = {
            "count": len(subset),
            "avg_faithfulness": round(sum(r["faithfulness_score"] for r in subset) / len(subset), 4),
            "avg_correctness": round(sum(r["correctness_score"] for r in subset) / len(subset), 4),
        }

    output_path = "scripts/generation_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Full results saved to {output_path}")
    return output


if __name__ == "__main__":
    run_generation_evaluation()