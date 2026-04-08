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
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv(dotenv_path="../../.env")
client_llm    = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
console       = Console()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(
    name="day5", metadata={"hnsw:space": "cosine"}
)

def naive_retrieve(question, n=10):
    vec     = bi_encoder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=vec, n_results=n,
        include=["documents", "distances", "metadatas"]
    )
    return [{"chunk": d, "distance": dist, "source": m["source"], "rank": i+1}
            for i, (d, dist, m) in enumerate(zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            ))]

def hyde_retrieve(question, n=10):
    hyp = client_llm.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=200,
        system="Write a 2-sentence factual answer as if from a technical paper. No first person. Stay strictly on topic.",
        messages=[{"role": "user", "content": question}]
    ).content[0].text
    vec     = bi_encoder.encode([hyp]).tolist()
    results = collection.query(
        query_embeddings=vec, n_results=n,
        include=["documents", "distances", "metadatas"]
    )
    return [{"chunk": d, "distance": dist, "source": m["source"], "rank": i+1, "hypothesis": hyp}
            for i, (d, dist, m) in enumerate(zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            ))]

def rerank(question, candidates, top_k=3):
    pairs  = [[question, c["chunk"]] for c in candidates]
    scores = cross_encoder.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    for i, c in enumerate(reranked):
        c["reranked_rank"] = i + 1
    return reranked[:top_k]

def find_keyword_rank(chunks, keyword):
    """Find which rank position contains the keyword. Returns None if not found."""
    for c in chunks:
        if keyword.lower() in c["chunk"].lower():
            return c.get("rank", c.get("reranked_rank"))
    return None

# ── Eval set ──────────────────────────────────────────────────────────────────
QUESTIONS = [
    {"q": "What is scaled dot-product attention?",           "keyword": "dot-product"},
    {"q": "How does multi-head attention work?",             "keyword": "head"},
    {"q": "What is positional encoding?",                    "keyword": "position"},
    {"q": "What optimizer and parameters were used?",        "keyword": "Adam"},
    {"q": "What is the role of layer normalisation?",        "keyword": "normalization"},
    {"q": "How does the encoder-decoder architecture work?", "keyword": "encoder"},
    {"q": "What were the BLEU scores on translation tasks?", "keyword": "BLEU"},
    {"q": "Why does the model use residual connections?",    "keyword": "residual"},
]

if __name__ == "__main__":
    if collection.count() == 0:
        print("[red]Collection empty. Run ingest.py first.[/red]")
        sys.exit(1)

    console.print(Rule("[bold]Day 5 — Naive vs HyDE vs Reranked[/bold]"))
    print(f"Collection: [bold]{collection.count()}[/bold] chunks\n")

    # ── Table 1: Hit rate summary ─────────────────────────────────────────────
    console.print("[bold]Table 1 — Retrieval hit rate (does top-3 contain the right chunk?)[/bold]")
    print("[dim]YES = keyword found in top-3 results · NO = missed · rank shown when all hit[/dim]\n")

    t1 = Table(show_header=True, header_style="bold", show_lines=True)
    t1.add_column("Question",                    width=38)
    t1.add_column("Keyword",                     width=13)
    t1.add_column("Naive\ntop-3 hit",            width=11)
    t1.add_column("HyDE\ntop-3 hit",             width=10)
    t1.add_column("Reranked\ntop-3 hit",         width=12)
    t1.add_column("Verdict",                     width=22)

    naive_hits = hyde_hits = rerank_hits = 0
    all_results = []

    for item in QUESTIONS:
        q       = item["q"]
        keyword = item["keyword"]

        naive_chunks  = naive_retrieve(q, n=10)
        hyde_chunks   = hyde_retrieve(q, n=10)

        # Reranker works on naive candidates (wide retrieval)
        rerank_chunks = rerank(q, naive_retrieve(q, n=10), top_k=3)

        # Check top-3 only for hit rate
        naive_top3  = naive_chunks[:3]
        hyde_top3   = hyde_chunks[:3]

        naive_hit   = any(keyword.lower() in c["chunk"].lower() for c in naive_top3)
        hyde_hit    = any(keyword.lower() in c["chunk"].lower() for c in hyde_chunks[:3])
        rerank_hit  = any(keyword.lower() in c["chunk"].lower() for c in rerank_chunks)

        # Find rank position of correct chunk in each method
        naive_rank  = find_keyword_rank(naive_chunks[:3],  keyword)
        hyde_rank   = find_keyword_rank(hyde_chunks[:3],   keyword)
        rerank_rank = find_keyword_rank(rerank_chunks,     keyword)

        if naive_hit:  naive_hits  += 1
        if hyde_hit:   hyde_hits   += 1
        if rerank_hit: rerank_hits += 1

        # Verdict — only declare a winner when methods differ
        all_hit  = naive_hit and hyde_hit and rerank_hit
        none_hit = not naive_hit and not hyde_hit and not rerank_hit

        if all_hit:
            verdict = "[dim]All retrieve correctly[/dim]"
        elif none_hit:
            verdict = "[red]All miss — corpus gap[/red]"
        elif rerank_hit and not naive_hit and not hyde_hit:
            verdict = "[green]Reranker recovers miss[/green]"
        elif rerank_hit and not naive_hit:
            verdict = "[green]Reranker + HyDE win[/green]"
        elif hyde_hit and not naive_hit and not rerank_hit:
            verdict = "[blue]HyDE only[/blue]"
        elif naive_hit and not hyde_hit and not rerank_hit:
            verdict = "[cyan]Naive only[/cyan]"
        else:
            verdict = "Mixed"

        t1.add_row(
            q[:38],
            keyword,
            f"[green]YES (rank {naive_rank})[/green]"  if naive_hit  else "[red]NO[/red]",
            f"[green]YES (rank {hyde_rank})[/green]"   if hyde_hit   else "[red]NO[/red]",
            f"[green]YES (rank {rerank_rank})[/green]" if rerank_hit else "[red]NO[/red]",
            verdict,
        )

        all_results.append({
            "q": q, "keyword": keyword,
            "naive_chunks": naive_chunks[:3],
            "hyde_chunks":  hyde_chunks[:3],
            "rerank_chunks": rerank_chunks,
            "naive_hit": naive_hit, "hyde_hit": hyde_hit, "rerank_hit": rerank_hit,
        })

    console.print(t1)

    n = len(QUESTIONS)
    print(f"\n[bold]Hit rate summary:[/bold]")
    print(f"  Naive RAG  : {naive_hits}/{n} questions retrieved correctly in top-3")
    print(f"  HyDE       : {hyde_hits}/{n} questions retrieved correctly in top-3")
    print(f"  Reranked   : {rerank_hits}/{n} questions retrieved correctly in top-3")

    # ── Table 2: Rank movement — the key reranking insight ───────────────────
    print()
    console.print(Rule())
    console.print("\n[bold]Table 2 — Rank movement after reranking[/bold]")
    print("[dim]Shows where the correct chunk ranked BEFORE reranking vs AFTER.[/dim]")
    print("[dim]A jump from rank 5 → rank 1 is the reranker doing its job.[/dim]\n")

    t2 = Table(show_header=True, header_style="bold", show_lines=True)
    t2.add_column("Question",                    width=38)
    t2.add_column("Keyword",                     width=13)
    t2.add_column("Rank before\nreranking",      width=14)
    t2.add_column("Rank after\nreranking",       width=13)
    t2.add_column("Rerank\nscore",               width=10)
    t2.add_column("Movement",                    width=16)

    for res in all_results:
        keyword = res["keyword"]

        # Find rank of keyword chunk in original wide retrieval (top-10)
        naive_wide = naive_retrieve(res["q"], n=10)
        rank_before = find_keyword_rank(naive_wide, keyword)

        # Find rank in reranked top-3
        rank_after  = find_keyword_rank(res["rerank_chunks"], keyword)

        # Find rerank score of the correct chunk
        rerank_score = None
        for c in res["rerank_chunks"]:
            if keyword.lower() in c["chunk"].lower():
                rerank_score = c.get("rerank_score")
                break

        if rank_before is None:
            movement = "[red]Not in top-10[/red]"
            rank_before_str = "Not found"
        elif rank_after is None:
            movement = "[red]Dropped out[/red]"
            rank_before_str = f"#{rank_before}"
        elif rank_before > rank_after:
            jump = rank_before - rank_after
            movement = f"[green]↑ +{jump} positions[/green]"
            rank_before_str = f"#{rank_before}"
        elif rank_before == rank_after:
            movement = "[dim]No change[/dim]"
            rank_before_str = f"#{rank_before}"
        else:
            drop = rank_after - rank_before
            movement = f"[yellow]↓ -{drop} positions[/yellow]"
            rank_before_str = f"#{rank_before}"

        score_str = f"{rerank_score:+.2f}" if rerank_score is not None else "—"

        t2.add_row(
            res["q"][:38],
            keyword,
            rank_before_str,
            f"#{rank_after}" if rank_after else "Not in top-3",
            score_str,
            movement,
        )

    console.print(t2)

    # ── Final scorecard ───────────────────────────────────────────────────────
    print()
    console.print(Rule())
    print(f"\n[bold]Phase progression:[/bold]")
    print(f"  Phase 1 baseline (naive, easy 6Q)  : 6/6   = 100%")
    print(f"  Day 5 naive      (harder 8Q)        : {naive_hits}/{n}")
    print(f"  Day 5 HyDE       (harder 8Q)        : {hyde_hits}/{n}")
    print(f"  Day 5 reranked   (harder 8Q)        : {rerank_hits}/{n}")

    if rerank_hits >= naive_hits:
        gained = rerank_hits - naive_hits
        print(f"\n[green]Reranking held or improved retrieval.[/green]")
        if gained > 0:
            print(f"[green]Recovered {gained} question(s) that naive RAG missed.[/green]")
    print("\n[dim]Next: Day 6 — RAGAS evaluation. Measure answer quality, not just retrieval.[/dim]")