import os
import sys
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from rich.console import Console

load_dotenv(dotenv_path="../../.env")
client_llm  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
console     = Console()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(
    name="day4",
    metadata={"hnsw:space": "cosine"}
)

def naive_retrieve(question: str, n: int = 3) -> list[dict]:
    vec     = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=vec, n_results=n,
                               include=["documents","distances","metadatas"])
    return [{"chunk": c, "dist": d, "src": m["source"]}
            for c, d, m in zip(results["documents"][0],
                                results["distances"][0],
                                results["metadatas"][0])]

def hyde_retrieve(question: str, n: int = 3) -> tuple[list[dict], str]:
    hyp = client_llm.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=200,
        system="Write a 2-sentence factual answer as if from a technical paper. No first person.",
        messages=[{"role": "user", "content": question}]
    ).content[0].text

    vec     = embed_model.encode([hyp]).tolist()
    results = collection.query(query_embeddings=vec, n_results=n,
                               include=["documents","distances","metadatas"])
    chunks  = [{"chunk": c, "dist": d, "src": m["source"]}
               for c, d, m in zip(results["documents"][0],
                                   results["distances"][0],
                                   results["metadatas"][0])]
    return chunks, hyp


QUESTIONS = [
    {"q": "What is scaled dot-product attention?",     "keyword": "dot-product"},
    {"q": "How does multi-head attention work?",       "keyword": "head"},
    {"q": "What is positional encoding?",              "keyword": "position"},
    {"q": "What optimizer was used for training?",     "keyword": "Adam"},
    {"q": "What is the role of layer normalisation?",  "keyword": "normalization"},
    {"q": "How does the encoder-decoder work?",        "keyword": "encoder"},
]

if __name__ == "__main__":
    if collection.count() == 0:
        print("[red]Collection empty. Run ingest.py first.[/red]")
        sys.exit(1)

    print(f"\n[bold]Naive RAG vs HyDE — {len(QUESTIONS)} questions[/bold]")
    print(f"Collection: {collection.count()} chunks\n")

    naive_hits = 0
    hyde_hits  = 0

    table = Table(show_header=True, header_style="bold")
    table.add_column("Question",       width=36)
    table.add_column("Keyword",        width=14)
    table.add_column("Naive",          width=7)
    table.add_column("Naive dist",     width=11)
    table.add_column("HyDE",           width=7)
    table.add_column("HyDE dist",      width=10)
    table.add_column("Winner",         width=8)

    for item in QUESTIONS:
        q       = item["q"]
        keyword = item["keyword"].lower()

        naive_chunks        = naive_retrieve(q)
        hyde_chunks, hyp    = hyde_retrieve(q)

        naive_text = " ".join(c["chunk"] for c in naive_chunks).lower()
        hyde_text  = " ".join(c["chunk"] for c in hyde_chunks).lower()

        naive_hit  = keyword in naive_text
        hyde_hit   = keyword in hyde_text

        naive_dist = naive_chunks[0]["dist"]
        hyde_dist  = hyde_chunks[0]["dist"]

        if naive_hit: naive_hits += 1
        if hyde_hit:  hyde_hits  += 1

        winner = ""
        # Determine winner honestly
        if hyde_hit and not naive_hit:
            winner = "[green]HyDE[/green]"
        elif naive_hit and not hyde_hit:
            winner = "[blue]Naive[/blue]"
        elif not naive_hit and not hyde_hit:
            winner = "[red]Both miss[/red]"
        else:
            # Both hit — compare by distance, with meaningful threshold
            diff = naive_dist - hyde_dist
            if diff > 0.02:        # HyDE meaningfully closer
                winner = "[green]HyDE[/green]"
            elif diff < -0.02:     # Naive meaningfully closer
                winner = "[blue]Naive[/blue]"
            else:
                winner = "Tie"     # Genuinely too close to call

        table.add_row(
            q[:36],
            item["keyword"],
            "[green]YES[/green]" if naive_hit else "[red]NO[/red]",
            f"{naive_dist:.4f}",
            "[green]YES[/green]" if hyde_hit  else "[red]NO[/red]",
            f"{hyde_dist:.4f}",
            winner,
        )

    console.print(table)
    print(f"\n[bold]Naive RAG hit rate : {naive_hits}/{len(QUESTIONS)}[/bold]")
    print(f"[bold]HyDE hit rate      : {hyde_hits}/{len(QUESTIONS)}[/bold]")

    if hyde_hits > naive_hits:
        print(f"\n[green]HyDE improved retrieval by {hyde_hits - naive_hits} question(s).[/green]")
    elif hyde_hits == naive_hits:
        print(f"\n[yellow]Same hit rate — check distance scores. Lower = better precision.[/yellow]")
    else:
        print(f"\n[red]Naive RAG won. Check your hypothetical answer quality.[/red]")

    print("\n[bold]Phase 1 baseline  : 6/6 on easy eval set[/bold]")
    print(f"[bold]Day 4 naive score : {naive_hits}/{len(QUESTIONS)} on harder questions[/bold]")
    print(f"[bold]Day 4 HyDE score  : {hyde_hits}/{len(QUESTIONS)} on harder questions[/bold]")