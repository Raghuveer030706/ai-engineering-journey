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
    name="day8", metadata={"hnsw:space": "cosine"}
)

# ── Step 1: Generate query expansions ────────────────────────────────────────
def expand_query(question: str, n_expansions: int = 3) -> list[str]:
    """
    Generate n reformulations of the question using different vocabulary.
    Each expansion is a different angle on the same information need.
    """
    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="""You are a search query expert. Given a question,
generate reformulations that use different vocabulary and phrasing
but seek the same information. Each reformulation should:
- Use different technical terms where possible
- Vary the sentence structure
- Stay strictly on the same topic
Output ONLY the reformulations, one per line, no numbering,
no explanation, no preamble.""",
        messages=[{
            "role": "user",
            "content": f"Generate {n_expansions} reformulations of this question:\n{question}"
        }]
    )
    expansions = [
        line.strip()
        for line in response.content[0].text.strip().split("\n")
        if line.strip() and len(line.strip()) > 10
    ]
    # Always include the original question
    all_queries = [question] + expansions[:n_expansions]
    return all_queries

# ── Step 2: Retrieve for each query ──────────────────────────────────────────
def retrieve_expanded(queries: list[str], n: int = 5) -> list[dict]:
    """
    Retrieve top-n candidates for each query.
    Tag each chunk with which query found it.
    """
    all_chunks = []

    for query in queries:
        vec     = bi_encoder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=vec, n_results=n,
            include=["documents", "distances", "metadatas"]
        )
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            all_chunks.append({
                "chunk":    doc,
                "distance": dist,
                "source":   meta["source"],
                "query":    query,
            })

    return all_chunks

# ── Step 3: Merge and deduplicate ─────────────────────────────────────────────
def merge_candidates(chunks: list[dict]) -> list[dict]:
    """
    Deduplicate by chunk text.
    Track how many different queries found each chunk —
    chunks found by multiple queries are stronger candidates.
    """
    seen   = {}
    merged = []

    for chunk in chunks:
        text = chunk["chunk"].strip()
        if text not in seen:
            seen[text] = len(merged)
            merged.append({
                **chunk,
                "found_by_queries": 1,
                "min_distance":     chunk["distance"],
            })
        else:
            idx = seen[text]
            merged[idx]["found_by_queries"] += 1
            merged[idx]["min_distance"] = min(
                merged[idx]["min_distance"], chunk["distance"]
            )

    # Sort by found_by_queries desc then distance asc before reranking
    # Chunks found by multiple queries bubble up as pre-rerank signal
    merged.sort(key=lambda x: (-x["found_by_queries"], x["min_distance"]))
    return merged

# ── Step 4: Rerank ────────────────────────────────────────────────────────────
def rerank(
    original_question: str,
    candidates: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    Rerank using the ORIGINAL question only.
    Expansion was for retrieval diversity — not for scoring.
    """
    pairs  = [[original_question, c["chunk"]] for c in candidates]
    scores = cross_encoder.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    reranked = sorted(
        candidates, key=lambda x: x["rerank_score"], reverse=True
    )
    filtered = [c for c in reranked if c["rerank_score"] > -2.0]
    return (filtered if filtered else reranked)[:top_k]

# ── Full pipeline ─────────────────────────────────────────────────────────────
def expanded_rag(
    question:     str,
    n_expansions: int = 3,
    n_per_query:  int = 5,
    top_k:        int = 5,
) -> dict:
    """
    Full query expansion pipeline:
    1. Generate n_expansions reformulations
    2. Retrieve n_per_query candidates per query
    3. Merge + deduplicate
    4. Rerank with original question
    5. Generate grounded answer from top_k
    """
    # Step 1: expand
    queries = expand_query(question, n_expansions)

    # Step 2+3: retrieve and merge
    all_candidates = retrieve_expanded(queries, n=n_per_query)
    candidates     = merge_candidates(all_candidates)

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
context provided. Be specific — include exact numbers, parameters,
and technical terms when the question asks for them.
If context is insufficient, say so clearly.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )

    return {
        "question":       question,
        "queries":        queries,
        "total_raw":      len(all_candidates),
        "after_dedup":    len(candidates),
        "top_chunks":     top_chunks,
        "answer":         response.content[0].text,
    }

# ── Display ───────────────────────────────────────────────────────────────────
def show_result(result: dict):
    multi_query_hits = sum(
        1 for c in result["top_chunks"]
        if c.get("found_by_queries", 1) > 1
    )

    expansions_str = "\n".join([
        f"  [{i+1}] {q}"
        for i, q in enumerate(result["queries"])
    ])

    print(Panel(
        f"[bold magenta]Question:[/bold magenta] {result['question']}\n\n"
        f"[bold cyan]Expanded queries ({len(result['queries'])} total):[/bold cyan]\n"
        f"{expansions_str}\n\n"
        f"[bold cyan]Retrieval pool:[/bold cyan]\n"
        f"  Raw candidates      : {result['total_raw']}\n"
        f"  After dedup         : {result['after_dedup']}\n"
        f"  Found by 2+ queries : {multi_query_hits} chunk(s) — strongest signal\n\n"
        f"[bold yellow]Top {len(result['top_chunks'])} after reranking:[/bold yellow]\n"
        + "\n".join([
            f"  [{i+1}] score={c['rerank_score']:+.2f}  "
            f"found_by={c.get('found_by_queries',1)} quer(ies)  "
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

    print(f"[bold]Query expansion RAG — {collection.count()} chunks[/bold]\n")

    QUESTIONS = [
        "What is positional encoding and why does the Transformer need it?",
        "What optimizer and exact parameters were used to train the Transformer?",
        "How does multi-head attention differ from single-head attention?",
        "What were the BLEU scores achieved on translation tasks?",
    ]

    for q in QUESTIONS:
        result = expanded_rag(q)
        show_result(result)
        print()