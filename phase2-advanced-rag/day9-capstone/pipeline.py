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
from rich.console import Console

load_dotenv(dotenv_path="../../.env")
client_llm    = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
console       = Console()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(
    name="day9", metadata={"hnsw:space": "cosine"}
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
    return [
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
    ], hyp

# ── Selective query expansion ─────────────────────────────────────────────────
def needs_expansion(naive_chunks: list[dict], threshold: float = 0.35) -> bool:
    """
    Only expand queries when naive retrieval is uncertain.
    If top chunk distance > threshold, retrieval is weak — expand.
    If top chunk distance <= threshold, retrieval is confident — skip expansion.
    This preserves Day 7 precision while recovering Day 8 recall gains.
    """
    if not naive_chunks:
        return True
    return naive_chunks[0]["distance"] > threshold

def expand_query(question: str, n_expansions: int = 2) -> list[str]:
    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="""Generate reformulations that use different vocabulary
but seek the same information. Output ONLY the reformulations,
one per line, no numbering, no explanation.""",
        messages=[{
            "role": "user",
            "content": f"Generate {n_expansions} reformulations:\n{question}"
        }]
    )
    expansions = [
        line.strip()
        for line in response.content[0].text.strip().split("\n")
        if line.strip() and len(line.strip()) > 10
    ]
    return expansions[:n_expansions]

def expanded_retrieve(question: str, n: int = 5) -> list[dict]:
    expansions = expand_query(question)
    chunks = []
    for exp in expansions:
        vec     = bi_encoder.encode([exp]).tolist()
        results = collection.query(
            query_embeddings=vec, n_results=n,
            include=["documents", "distances", "metadatas"]
        )
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            chunks.append({
                "chunk":    doc,
                "distance": dist,
                "source":   meta["source"],
                "method":   "expanded",
            })
    return chunks

# ── Merge and deduplicate ─────────────────────────────────────────────────────
def merge_candidates(chunk_lists: list[list[dict]]) -> list[dict]:
    seen   = {}
    merged = []
    for chunks in chunk_lists:
        for chunk in chunks:
            text = chunk["chunk"].strip()
            if text not in seen:
                seen[text] = len(merged)
                merged.append({
                    **chunk,
                    "found_by": [chunk["method"]],
                    "min_distance": chunk["distance"],
                })
            else:
                idx = seen[text]
                merged[idx]["found_by"].append(chunk["method"])
                merged[idx]["min_distance"] = min(
                    merged[idx]["min_distance"],
                    chunk["distance"]
                )
    return merged

# ── Rerank ────────────────────────────────────────────────────────────────────
def rerank(
    question:   str,
    candidates: list[dict],
    top_k:      int = 5
) -> list[dict]:
    pairs  = [[question, c["chunk"]] for c in candidates]
    scores = cross_encoder.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    reranked = sorted(
        candidates, key=lambda x: x["rerank_score"], reverse=True
    )
    filtered = [c for c in reranked if c["rerank_score"] > -2.0]
    return (filtered if filtered else reranked)[:top_k]

# ── Full Phase 2 pipeline ─────────────────────────────────────────────────────
def phase2_rag(question: str, n_wide: int = 10, top_k: int = 5) -> dict:
    """
    Complete Phase 2 pipeline:
    1. Naive retrieval (always)
    2. HyDE retrieval  (always — best recall/precision combo from Day 7)
    3. Selective expansion (only if naive top chunk distance > 0.35)
    4. Merge all candidate pools
    5. Rerank with cross-encoder
    6. Generate grounded answer
    """
    # Step 1: Naive
    naive_chunks = naive_retrieve(question, n=n_wide)

    # Step 2: HyDE
    hyde_chunks, hypothesis = hyde_retrieve(question, n=n_wide)

    # Step 3: Selective expansion
    expansion_used = False
    all_pools = [naive_chunks, hyde_chunks]

    if needs_expansion(naive_chunks):
        expansion_used = True
        exp_chunks = expanded_retrieve(question, n=5)
        all_pools.append(exp_chunks)

    # Step 4: Merge
    candidates = merge_candidates(all_pools)

    # Step 5: Rerank
    top_chunks = rerank(question, candidates, top_k=top_k)

    # Step 6: Generate
    context = "\n\n".join([
        f"[{i+1}] (source: {c['source']})\n{c['chunk']}"
        for i, c in enumerate(top_chunks)
    ])
    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="""You are a helpful assistant. Answer using ONLY the
context provided. Be specific — include exact numbers, parameters,
and technical terms when the question asks for them.
If context is insufficient, say so clearly.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )

    # Chunks found by multiple methods — strongest signal
    multi_hit = sum(
        1 for c in top_chunks if len(c.get("found_by", [])) > 1
    )

    return {
        "question":       question,
        "hypothesis":     hypothesis,
        "expansion_used": expansion_used,
        "naive_count":    len(naive_chunks),
        "hyde_count":     len(hyde_chunks),
        "merged":         len(candidates),
        "top_chunks":     top_chunks,
        "multi_hit":      multi_hit,
        "answer":         response.content[0].text,
    }

# ── Display ───────────────────────────────────────────────────────────────────
def show_result(result: dict):
    expansion_note = (
        "[yellow]expansion used (weak naive signal)[/yellow]"
        if result["expansion_used"]
        else "[dim]no expansion needed (strong naive signal)[/dim]"
    )

    print(Panel(
        f"[bold magenta]Question:[/bold magenta] {result['question']}\n\n"
        f"[bold cyan]Pipeline decisions:[/bold cyan]\n"
        f"  Naive candidates   : {result['naive_count']}\n"
        f"  HyDE candidates    : {result['hyde_count']}\n"
        f"  Expansion          : {expansion_note}\n"
        f"  After merge+dedup  : {result['merged']}\n"
        f"  Multi-method hits  : {result['multi_hit']} chunk(s)\n\n"
        f"[bold yellow]Top {len(result['top_chunks'])} after reranking:[/bold yellow]\n"
        + "\n".join([
            f"  [{i+1}] score={c['rerank_score']:+.2f}  "
            f"methods={'+'.join(c.get('found_by',['?']))}  "
            f"src={c['source']}\n"
            f"      {c['chunk'][:110]}..."
            for i, c in enumerate(result["top_chunks"])
        ])
        + f"\n\n[bold green]Answer:[/bold green]\n{result['answer']}",
        expand=False
    ))

if __name__ == "__main__":
    if collection.count() == 0:
        print("[red]Collection empty. Run ingest.py first.[/red]")
        sys.exit(1)

    print(f"[bold]Phase 2 capstone pipeline — {collection.count()} chunks[/bold]\n")

    QUESTIONS = [
        "What is scaled dot-product attention and how is it computed?",
        "What optimizer and exact parameters were used to train the Transformer?",
        "What is positional encoding and why does the Transformer need it?",
        "How does multi-head attention improve on single-head attention?",
    ]

    for q in QUESTIONS:
        result = phase2_rag(q)
        show_result(result)
        print()