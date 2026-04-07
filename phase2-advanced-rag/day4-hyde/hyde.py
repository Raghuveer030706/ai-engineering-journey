import os
import sys
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel
from rich.table import Table

load_dotenv(dotenv_path="../../.env")
client_llm  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(
    name="day4",
    metadata={"hnsw:space": "cosine"}
)

# ── Step 1: Generate hypothetical answer ──────────────────────────────────────
def generate_hypothetical_answer(question: str) -> str:
    """
    Ask Claude to write a hypothetical answer to the question.
    This answer is NOT grounded in your documents — it's used
    only to create a better search vector.
    """
    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="""You are an expert technical writer.
Write a concise, factual answer to the question as if it appeared
in a high-quality technical document or research paper.
Write 2-3 sentences only. Do not say 'I' or reference yourself.
Stay strictly on the specific topic asked — do not generalise.""",
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

# ── Step 2: Retrieve using hypothetical answer vector ─────────────────────────
def hyde_retrieve(question: str, n_results: int = 3) -> dict:
    """
    HyDE retrieval:
    1. Generate hypothetical answer
    2. Embed the answer (not the question)
    3. Search ChromaDB with that richer vector
    """
    hyp_answer = generate_hypothetical_answer(question)
    hyp_vec    = embed_model.encode([hyp_answer]).tolist()

    results = collection.query(
        query_embeddings=hyp_vec,
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )

    return {
        "question":         question,
        "hypothesis":       hyp_answer,
        "chunks":           results["documents"][0],
        "distances":        results["distances"][0],
        "sources":          [m["source"] for m in results["metadatas"][0]],
    }

# ── Step 3: Generate grounded answer ─────────────────────────────────────────
def hyde_rag(question: str, n_results: int = 3) -> dict:
    """Full HyDE RAG: hypothetical retrieval + grounded generation."""
    retrieved = hyde_retrieve(question, n_results)

    context = "\n\n".join([
        f"[{i+1}] (source: {src})\n{chunk}"
        for i, (chunk, src) in enumerate(
            zip(retrieved["chunks"], retrieved["sources"])
        )
    ])

    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="""You are a helpful assistant. Answer using ONLY the context
provided. If context is insufficient, say so clearly.""",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )

    retrieved["answer"] = response.content[0].text
    return retrieved


# ── Display results ───────────────────────────────────────────────────────────
def show_result(result: dict):
    print(Panel(
        f"[bold magenta]Question:[/bold magenta] {result['question']}\n\n"
        f"[bold cyan]Hypothetical answer used for retrieval:[/bold cyan]\n"
        f"  {result['hypothesis']}\n\n"
        f"[bold yellow]Retrieved chunks:[/bold yellow]\n"
        + "\n".join([
            f"  [{i+1}] (dist={d:.4f}, src={s})\n      {c[:120]}..."
            for i, (c, d, s) in enumerate(zip(
                result["chunks"], result["distances"], result["sources"]
            ))
        ])
        + f"\n\n[bold green]Grounded answer:[/bold green]\n{result['answer']}",
        expand=False
    ))


if __name__ == "__main__":
    if collection.count() == 0:
        print("[red]Collection empty. Run ingest.py first.[/red]")
        sys.exit(1)

    print(f"[bold]Collection has {collection.count()} chunks[/bold]\n")

    TEST_QUESTIONS = [
        "What is the attention mechanism and why was it introduced?",
        "How does multi-head attention differ from single-head attention?",
        "What training techniques were used to train the Transformer?",
        "What does positional encoding do and why is it needed?",
    ]

    for q in TEST_QUESTIONS:
        result = hyde_rag(q)
        show_result(result)
        print()