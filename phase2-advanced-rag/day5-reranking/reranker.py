import os
import sys
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
import anthropic
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from rich.panel import Panel

load_dotenv(dotenv_path="../../.env")
client_llm  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Bi-encoder — same as before, fast retrieval
bi_encoder  = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-encoder — new today, precise reranking
# Reads question + chunk together, outputs a relevance score
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(
    name="day5",
    metadata={"hnsw:space": "cosine"}
)

# ── Step 1: Wide retrieval with bi-encoder ────────────────────────────────────
def retrieve_wide(question: str, n: int = 10) -> list[dict]:
    """
    Retrieve top-n candidates by vector distance.
    Cast a wide net — reranker will filter down to the best.
    """
    vec     = bi_encoder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=vec,
        n_results=n,
        include=["documents", "distances", "metadatas"]
    )
    return [
        {
            "chunk":    doc,
            "distance": dist,
            "source":   meta["source"],
        }
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ]

# ── Step 2: Rerank with cross-encoder ────────────────────────────────────────
def rerank(question: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Score each candidate by reading question + chunk together.
    Returns top_k by relevance score (higher = more relevant).
    """
    # Cross-encoder expects list of [question, chunk] pairs
    pairs  = [[question, c["chunk"]] for c in candidates]
    scores = cross_encoder.predict(pairs)

    # Attach scores to candidates
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    # Sort by score descending — highest relevance first
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

# ── Step 3: Full reranked RAG pipeline ───────────────────────────────────────
def reranked_rag(question: str) -> dict:
    """
    Full pipeline:
    retrieve top-10 → rerank → generate answer from top-3
    """
    # Wide retrieval
    candidates = retrieve_wide(question, n=10)

    # Rerank
    top_chunks = rerank(question, candidates, top_k=3)

    # Build grounded prompt
    context = "\n\n".join([
        f"[{i+1}] (source: {c['source']})\n{c['chunk']}"
        for i, c in enumerate(top_chunks)
    ])

    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="""You are a helpful assistant. Answer using ONLY the
context provided. If context is insufficient, say so clearly.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )

    return {
        "question":   question,
        "candidates": candidates,
        "top_chunks": top_chunks,
        "answer":     response.content[0].text,
    }

# ── Display ───────────────────────────────────────────────────────────────────
def show_result(result: dict):
    print(Panel(
        f"[bold magenta]Question:[/bold magenta] {result['question']}\n\n"
        f"[bold yellow]Top 3 after reranking:[/bold yellow]\n"
        + "\n".join([
            f"  [{i+1}] score={c['rerank_score']:+.4f}  dist={c['distance']:.4f}  "
            f"src={c['source']}\n      {c['chunk'][:120]}..."
            for i, c in enumerate(result["top_chunks"])
        ])
        + f"\n\n[bold green]Answer:[/bold green]\n{result['answer']}",
        expand=False
    ))

if __name__ == "__main__":
    if collection.count() == 0:
        print("[red]Collection empty. Run ingest.py first.[/red]")
        sys.exit(1)

    print(f"[bold]Reranked RAG — {collection.count()} chunks[/bold]\n")

    QUESTIONS = [
        "What is the role of layer normalisation in the Transformer?",
        "What optimizer was used and what were the specific parameters?",
        "How does multi-head attention improve on single-head attention?",
        "What is the intuition behind positional encoding?",
    ]

    for q in QUESTIONS:
        result = reranked_rag(q)
        show_result(result)
        print()