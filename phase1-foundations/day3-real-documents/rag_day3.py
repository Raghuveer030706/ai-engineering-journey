import os
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from rich.panel import Panel

load_dotenv(dotenv_path="../../.env")

client_llm  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(
    name="day3",
    metadata={"hnsw:space": "cosine"}
)

def rag(question: str, n_results: int = 3) -> dict:
    query_vec = embed_model.encode([question]).tolist()
    results   = collection.query(
        query_embeddings=query_vec,
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )

    chunks    = results["documents"][0]
    distances = results["distances"][0]
    sources   = [m["source"] for m in results["metadatas"][0]]

    # Build context with source labels
    context_parts = []
    for i, (chunk, source) in enumerate(zip(chunks, sources)):
        context_parts.append(f"[{i+1}] (source: {source})\n{chunk}")
    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful assistant. Answer using ONLY the
    context provided. Each chunk is labeled with its source.
    When chunks from different sources conflict, prefer the academic paper
    over personal notes. If the context is insufficient, say so clearly."""

    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        system=system_prompt,
    )

    return {
        "question":  question,
        "chunks":    chunks,
        "distances": distances,
        "sources":   sources,
        "answer":    response.content[0].text,
    }

def run_custom_eval():
    EVAL_SET = [
        {"question": "What is Scaled Dot-Product Attention?",
         "expected_keyword": "dot-product"},
        {"question": "How many attention heads did the best model use?",
         "expected_keyword": "heads"},
        {"question": "How long did training take on P100 GPUs?",
         "expected_keyword": "3.5 days"},
        {"question": "What optimization algorithm was used?",
         "expected_keyword": "Adam"},
        {"question": "What does BLEU score measure?",
         "expected_keyword": "BLEU"},
        {"question": "What is the role of positional encoding?",
         "expected_keyword": "position"},
    ]

    print("\n[bold]--- Day 3 Custom Eval ---[/bold]\n")
    print(f"Running {len(EVAL_SET)} questions against {collection.count()} real chunks\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Question",         width=40)
    table.add_column("Expected keyword", width=18)
    table.add_column("Pass",             width=6)
    table.add_column("Top source",       width=20)

    passed = 0
    for item in EVAL_SET:
        result   = rag(item["question"], n_results=3)
        all_text = " ".join(result["chunks"]).lower()
        hit      = item["expected_keyword"].lower() in all_text

        if hit:
            passed += 1

        table.add_row(
            item["question"][:40],
            item["expected_keyword"],
            "[green]YES[/green]" if hit else "[red]NO[/red]",
            result["sources"][0],
        )

    print(table)

    score = passed / len(EVAL_SET) * 100
    print(f"\n[bold]Retrieval hit rate: {passed}/{len(EVAL_SET)} = {score:.0f}%[/bold]")
    print(f"\nDay 2 baseline (toy corpus) : 8/8 = 100%")
    print(f"Day 3 result  (real docs)   : {passed}/{len(EVAL_SET)} = {score:.0f}%")

    if score < 100:
        print("\n[yellow]Some questions missed. These are your Phase 2 targets.[/yellow]")
        print("HyDE and reranking will recover these in Week 3.")
    else:
        print("\n[green]Clean sweep on real documents. Strong foundation.[/green]")

def run_eval():
    """
    Build eval questions FROM your actual documents.
    After running ingest.py, read a chunk of your document
    and write 5 questions whose answers are clearly in it.
    This is manual but honest -- no fake eval sets.
    """
    print("[bold yellow]First, run ingest.py and read your documents.[/bold yellow]")
    print("Then replace the EVAL_SET below with questions from YOUR content.\n")

    # Placeholder -- replace with questions from your actual documents
    EVAL_SET = [
        {"question": "What is the main topic of the first document?", "expected_keyword": ""},
    ]

    print("[bold]--- Day 3 RAG Eval ---[/bold]\n")
    print(f"Collection has [bold]{collection.count()}[/bold] chunks from real documents.\n")

    # Show 3 sample questions with full output
    SAMPLE_QUESTIONS = [
        "What are the key concepts discussed in the documents?",
        "Summarize the main argument of the text.",
        "What technical terms are introduced?",
    ]

    for q in SAMPLE_QUESTIONS:
        result = rag(q)
        print(Panel(
            f"[bold magenta]Q:[/bold magenta] {result['question']}\n\n"
            f"[bold yellow]Sources:[/bold yellow] {', '.join(set(result['sources']))}\n"
            f"[bold yellow]Top chunk (dist={result['distances'][0]:.4f}):[/bold yellow]\n"
            f"  {result['chunks'][0][:200]}\n\n"
            f"[bold green]Answer:[/bold green] {result['answer']}",
            expand=False
        ))
        print()


if __name__ == "__main__":
    if collection.count() == 0:
        print("[red]Collection is empty. Run ingest.py first.[/red]")
    else:
        run_eval()
        run_custom_eval()