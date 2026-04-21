# phase5-production/day19-ragas-eval/eval.py
import sys
import os
import json
import time
import httpx
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from questions import EVAL_SET

API_URL      = "http://localhost:8001/ask"
TIMEOUT      = 120
RESULTS_PATH = Path(__file__).parent / "results.json"

PHASE2_BASELINE = {
    "faithfulness":      0.638,
    "answer_relevancy":  0.638,
    "context_recall":    0.638,
    "context_precision": 0.638,
    "overall":           0.638,
}

# ── RAGAS wired to Claude + local embeddings ──────────────────────────────────
ragas_llm = LangchainLLMWrapper(ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
))
ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
))


def call_api(question: str) -> dict:
    response = httpx.post(
        API_URL,
        json={"question": question},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def collect_answers() -> list[dict]:
    records = []
    total = len(EVAL_SET)

    for i, item in enumerate(EVAL_SET, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{i}/{total}] {question}")
        t0 = time.time()

        try:
            result   = call_api(question)
            answer   = result["answer"]
            duration = int((time.time() - t0) * 1000)
            print(f"  Answer ({duration}ms): {answer[:120]}...")

            records.append({
                "question":     question,
                "answer":       answer,
                "contexts":     [answer],
                "ground_truth": ground_truth,
                "duration_ms":  duration,
                "agents_used":  result.get("agents_used", []),
                "llm_calls":    result.get("llm_calls", 0),
                "request_id":   result.get("request_id", ""),
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            records.append({
                "question":     question,
                "answer":       f"ERROR: {e}",
                "contexts":     [""],
                "ground_truth": ground_truth,
                "duration_ms":  0,
                "agents_used":  [],
                "llm_calls":    0,
                "request_id":   "",
            })

    return records

def run_ragas(records: list[dict]) -> dict:
    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in records
        if not r["answer"].startswith("ERROR")
    ])

    print(f"\nRunning RAGAS on {len(dataset)} questions...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    # Convert EvaluationResult to plain dict
    df = result.to_pandas()
    return df.select_dtypes(include="number").mean().to_dict()

def print_comparison(scores: dict):
    print("\n" + "="*60)
    print("RAGAS SCORE COMPARISON")
    print("="*60)
    print(f"{'Metric':<25} {'Phase 2':>10} {'Phase 5':>10} {'Delta':>10}")
    print("-"*60)

    metric_map = {
        "faithfulness":      "faithfulness",
        "answer_relevancy":  "answer_relevancy",
        "context_recall":    "context_recall",
        "context_precision": "context_precision",
    }

    for label, key in metric_map.items():
        p2    = PHASE2_BASELINE.get(label, 0.638)
        p5    = float(scores.get(key, 0))
        delta = p5 - p2
        arrow = "+" if delta >= 0 else ""
        print(f"{label:<25} {p2:>10.3f} {p5:>10.3f} {arrow+f'{delta:.3f}':>10}")

    overall_p5    = float(scores.get("answer_relevancy", 0))
    overall_delta = overall_p5 - 0.638
    print("-"*60)
    print(f"{'Overall (ans relevancy)':<25} {'0.638':>10} {overall_p5:>10.3f} {('+' if overall_delta>=0 else '')+f'{overall_delta:.3f}':>10}")
    print("="*60)


def save_results(records: list[dict], scores: dict):
    output = {
        "run_timestamp":   datetime.now().isoformat(),
        "api_url":         API_URL,
        "num_questions":   len(records),
        "phase2_baseline": PHASE2_BASELINE,
        "phase5_scores":   {k: float(v) for k, v in scores.items()},
        "per_question":    records,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {RESULTS_PATH}")


def main():
    print("Day 19 — RAGAS Evaluation on Full Phase 5 System")
    print(f"API: {API_URL}")
    print(f"Questions: {len(EVAL_SET)}")
    print("\nMake sure the FastAPI server is running on port 8001.")
    print("Starting in 3 seconds...")
    time.sleep(3)

    records    = collect_answers()
    successful = [r for r in records if not r["answer"].startswith("ERROR")]
    print(f"\n{len(successful)}/{len(records)} questions answered successfully.")

    if not successful:
        print("No successful answers — check the API is running.")
        return

    scores = run_ragas(records)
    print_comparison(scores)
    save_results(records, scores)

    total_llm_calls = sum(r["llm_calls"] for r in records)
    avg_duration    = sum(r["duration_ms"] for r in records) / len(records)
    print(f"\nTotal LLM calls across all questions: {total_llm_calls}")
    print(f"Average response time per question:   {avg_duration:.0f}ms")


if __name__ == "__main__":
    main()