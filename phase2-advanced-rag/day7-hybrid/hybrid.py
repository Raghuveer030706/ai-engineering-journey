import os
import sys
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.console import Console

load_dotenv(dotenv_path="../../.env")
client_llm    = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
console       = Console()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(
    name="day7", metadata={"hnsw:space": "cosine"}
)

# ── Naive retrieval ───────────────────────────────────────────────────────────
def naive_retrieve(question: str, n: int = 10) -> list[dict]:
    vec     = bi_encoder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=vec, n_results=n,
        include=["documents", "distances", "metadatas"]
    )
    return [
        {
            "chunk":    doc,
            "distance": dist,
            "source":   meta["source"],
            "method":   "naive",
        }
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ]

# ── HyDE retrieval ────────────────────────────────────────────────────────────
def hyde_retrieve(question: str, n: int = 10) -> tuple[list[dict], str]:
    hyp = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="""Write a 2-sentence factual answer as if from a
technical paper. No first person. Stay strictly on topic.""",
        messages=[{"role": "user", "content": question}]
    ).content[0].text

    vec     = bi_encoder.encode([hyp]).tolist()
    results = collection.query(
        query_embeddings=vec, n_results=n,
        include=["documents", "distances", "metadatas"]
    )
    chunks = [
        {
            "chunk":    doc,
            "distance": dist,
            "source":   meta["source"],
            "method":   "hyde",
        }
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )
    ]
    return chunks, hyp

# ── Merge and deduplicate ─────────────────────────────────────────────────────
def merge_candidates(
    naive_chunks: list[dict],
    hyde_chunks:  list[dict],
) -> list[dict]:
    """
    Combine both retrieval pools and deduplicate by chunk text.
    When the same chunk appears in both pools, keep it once
    and tag it as found by both methods — a strong signal.
    """
    seen   = {}
    merged = []

    for chunk in naive_chunks + hyde_chunks:
        text = chunk["chunk"].strip()

        if text not in seen:
            seen[text] = len(merged)
            merged.append({**chunk, "found_by": [chunk["method"]]})
        else:
            # Same chunk found by both — upgrade the tag
            idx = seen[text]
            merged[idx]["found_by"].append(chunk["method"])
            # Keep the better (lower) distance
            merged[idx]["distance"] = min(
                merged[idx]["distance"], chunk["distance"]
            )

    return merged

# ── Rerank combined pool ──────────────────────────────────────────────────────
def rerank(question: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    pairs  = [[question, c["chunk"]] for c in candidates]
    scores = cross_encoder.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    reranked = sorted(
        candidates, key=lambda x: x["rerank_score"], reverse=True
    )
    # New — only drop clearly irrelevant chunks (score below -2.0)
    # Mathematical notation and truncated formulas score around -0.3
    # that's different from truly irrelevant chunks which score -3.0 to -8.0
    filtered = [c for c in reranked if c["rerank_score"] > -2.0]
    return (filtered if filtered else reranked)[:top_k]

# ── Full hybrid pipeline ──────────────────────────────────────────────────────
def hybrid_rag(question: str, n_wide: int = 10, top_k: int = 5) -> dict:
    """
    Full hybrid pipeline:
    1. Naive retrieval (top n_wide)
    2. HyDE retrieval  (top n_wide)
    3. Merge + deduplicate
    4. Rerank combined pool
    5. Generate grounded answer from top_k
    """
    # Step 1 + 2: retrieve from both angles
    naive_chunks            = naive_retrieve(question, n=n_wide)
    hyde_chunks, hypothesis = hyde_retrieve(question,  n=n_wide)

    # Step 3: merge
    candidates = merge_candidates(naive_chunks, hyde_chunks)

    # Step 4: rerank
    top_chunks = rerank(question, candidates, top_k=top_k)

    # Step 5: generate
    context = "\n\n".join([
        f"[{i+1}] (source: {c['source']})\n{c['chunk']}"
        for i, c in enumerate(top_chunks)
    ])
    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="""You are a helpful assistant. Answer using ONLY the
context provided. Be specific — if the question asks for numbers
or parameters, include them explicitly.
If context is insufficient, say so clearly.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )

    return {
        "question":    question,
        "hypothesis":  hypothesis,
        "naive_count": len(naive_chunks),
        "hyde_count":  len(hyde_chunks),
        "merged":      len(candidates),
        "top_chunks":  top_chunks,
        "answer":      response.content[0].text,
    }

# ── Display ───────────────────────────────────────────────────────────────────
def show_result(result: dict):
    # Candidate pool summary
    both_count = sum(
        1 for c in result["top_chunks"]
        if len(c.get("found_by", [])) > 1
    )

    print(Panel(
        f"[bold magenta]Question:[/bold magenta] {result['question']}\n\n"
        f"[bold cyan]Retrieval pool:[/bold cyan]\n"
        f"  Naive candidates  : {result['naive_count']}\n"
        f"  HyDE candidates   : {result['hyde_count']}\n"
        f"  After merge+dedup : {result['merged']}\n"
        f"  Found by both     : {both_count} chunk(s) — strong signal\n\n"
        f"[bold yellow]Top {len(result['top_chunks'])} after reranking:[/bold yellow]\n"
        + "\n".join([
            f"  [{i+1}] score={c['rerank_score']:+.2f}  "
            f"found_by={'+'.join(c.get('found_by',['?']))}  "
            f"src={c['source']}\n"
            f"      {c['chunk'][:110]}..."
            for i, c in enumerate(result["top_chunks"])
        ])
        + f"\n\n[bold green]Answer:[/bold green]\n{result['answer']}",
        expand=False
    ))

# ── Test questions ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if collection.count() == 0:
        print("[red]Collection empty. Run ingest.py first.[/red]")
        sys.exit(1)

    print(f"[bold]Hybrid RAG — {collection.count()} chunks[/bold]\n")

    QUESTIONS = [
        "What is scaled dot-product attention and how is it computed?",
        "What optimizer and exact parameters were used to train the Transformer?",
        "How does multi-head attention improve on single-head attention?",
        "What is the role of layer normalisation and residual connections?",
    ]

    for q in QUESTIONS:
        result = hybrid_rag(q)
        show_result(result)
        print()