import os
import sys
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from rich.console import Console
from rich.rule import Rule

load_dotenv(dotenv_path="../../.env")

# ── Models ────────────────────────────────────────────────────────────────────
client_llm    = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
console       = Console()

# ── ChromaDB ──────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection    = chroma_client.get_or_create_collection(
    name="day6", metadata={"hnsw:space": "cosine"}
)

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(question: str, n_wide: int = 20, top_k: int = 5) -> list[str]:
    vec      = bi_encoder.encode([question]).tolist()
    results  = collection.query(
        query_embeddings=vec, n_results=n_wide,
        include=["documents", "distances"]
    )
    candidates = [{"chunk": d, "distance": dist}
                  for d, dist in zip(results["documents"][0],
                                     results["distances"][0])]
    pairs  = [[question, c["chunk"]] for c in candidates]
    scores = cross_encoder.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    filtered = [c for c in reranked if c["rerank_score"] > 0.0]
    return [c["chunk"] for c in (filtered if filtered else reranked)[:top_k]]

def generate_answer(question: str, contexts: list[str]) -> str:
    context = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="""You are a helpful assistant. Answer using ONLY the
context provided.Be specific and precise — if the question asks for specific
numbers, parameters, or values, include them explicitly in your answer. If context is insufficient, say so clearly.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text

# ── Eval set ──────────────────────────────────────────────────────────────────
EVAL_SET = [
    {
        "question": "What is scaled dot-product attention?",
        "ground_truth": "Scaled dot-product attention computes attention weights by taking the dot product of queries and keys, dividing by the square root of the key dimension, applying softmax, and using the result to weight the values."
    },
    {
        "question": "How does multi-head attention work?",
        "ground_truth": "Multi-head attention projects queries, keys and values h times with different learned projections, applies attention in parallel, then concatenates and projects the results allowing the model to attend to information from different representation subspaces."
    },
    {
        "question": "What is positional encoding?",
        "ground_truth": "Positional encoding adds fixed sinusoidal signals to input embeddings to give the model information about the relative or absolute position of tokens in the sequence since the model contains no recurrence or convolution."
    },
    {
        "question": "What optimizer and parameters were used to train the Transformer?",
        "ground_truth": "The Adam optimizer was used with beta1 of 0.9, beta2 of 0.98, and epsilon of 10 to the power of negative 9, along with a custom learning rate schedule that increases linearly for warmup steps then decreases proportionally."
    },
    {
        "question": "What is the role of layer normalisation in the Transformer?",
        "ground_truth": "Layer normalisation is applied around each sub-layer in the encoder and decoder using a residual connection followed by layer normalisation, stabilising training and allowing deeper networks."
    },
    {
        "question": "Why does the model use residual connections?",
        "ground_truth": "Residual connections are used around each sub-layer so that the output of each sub-layer is the layer normalisation of the sum of the sub-layer input and the sub-layer output, helping gradients flow during training."
    },
]

# ── Run pipeline ──────────────────────────────────────────────────────────────
def run_pipeline():
    print(f"\n[bold cyan]Running RAG pipeline on {len(EVAL_SET)} questions...[/bold cyan]\n")

    rows = []
    for item in EVAL_SET:
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"[dim]Retrieving:[/dim] {q[:60]}...")
        ctx = retrieve(q)
        ans = generate_answer(q, ctx)
        rows.append({
            "question":     q,
            "answer":       ans,
            "contexts":     ctx,
            "ground_truth": gt,
        })
        print(f"  [dim]Answer: {ans[:80]}...[/dim]")

    return rows

# ── Manual RAGAS 0.4.3 evaluation ────────────────────────────────────────────
def run_ragas_043(rows: list[dict]):
    """
    RAGAS 0.4.3 uses a different API from 0.5+.
    We build the dataset manually and call evaluate correctly.
    """
    print(f"\n[bold cyan]Running RAGAS 0.4.3 evaluation...[/bold cyan]")
    print("[dim]Makes LLM calls to score each metric — takes 2-3 minutes.[/dim]\n")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_anthropic import ChatAnthropic
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Build dataset in exact format RAGAS 0.4.3 expects
    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in rows],
        "answer":       [r["answer"]       for r in rows],
        "contexts":     [r["contexts"]     for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    # LLM wrapper — 0.4.3 specific
    llm = LangchainLLMWrapper(ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2048,
        temperature=0,
    ))

    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ))

    # Set LLM and embeddings on each metric — required in 0.4.3
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = emb

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
    )

    return result

# ── Display ───────────────────────────────────────────────────────────────────
def display_results(result, rows: list[dict]):
    console.print(Rule("[bold]RAGAS Evaluation Results — Day 6[/bold]"))

    # Build DataFrame first — this always works in 0.4.3
    try:
        df = result.to_pandas()
        df.columns = [str(c).lower().strip() for c in df.columns]

        # Inject question column if missing
        if "question" not in df.columns:
            df.insert(0, "question", [r["question"] for r in rows])

    except Exception as e:
        print(f"[red]Could not build DataFrame: {e}[/red]")
        return

    # ── Per-question table ────────────────────────────────────────────────────
    t = Table(show_header=True, header_style="bold", show_lines=True)
    t.add_column("Question",         width=36)
    t.add_column("Faithfulness",     width=13)
    t.add_column("Ans relevancy",    width=13)
    t.add_column("Ctx precision",    width=13)
    t.add_column("Ctx recall",       width=11)
    t.add_column("Verdict",          width=14)

    def fmt(v):
        try:
            v = float(v) if v is not None else 0.0
            if v >= 0.8:   return f"[green]{v:.2f}[/green]"
            elif v >= 0.6: return f"[yellow]{v:.2f}[/yellow]"
            else:          return f"[red]{v:.2f}[/red]"
        except Exception:
            return "[dim]n/a[/dim]"

    def safe_col(row, col):
        try:
            v = row.get(col, 0)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    for _, row in df.iterrows():
        fq  = safe_col(row, "faithfulness")
        arq = safe_col(row, "answer_relevancy")
        cpq = safe_col(row, "context_precision")
        crq = safe_col(row, "context_recall")
        avg = (fq + arq + cpq + crq) / 4

        verdict = "[green]Strong[/green]"       if avg >= 0.75 else \
                  "[yellow]Needs work[/yellow]" if avg >= 0.55 else \
                  "[red]Failing[/red]"

        t.add_row(
            str(row.get("question", ""))[:36],
            fmt(fq), fmt(arq), fmt(cpq), fmt(crq),
            verdict,
        )

    console.print(t)

    # ── Overall scores — computed from DataFrame averages ─────────────────────
    # This is the correct method for RAGAS 0.4.3
    def col_avg(col):
        try:
            if col in df.columns:
                vals = df[col].dropna().astype(float)
                return float(vals.mean()) if len(vals) > 0 else 0.0
            return 0.0
        except Exception:
            return 0.0

    f  = col_avg("faithfulness")
    ar = col_avg("answer_relevancy")
    cp = col_avg("context_precision")
    cr = col_avg("context_recall")
    avg_all = (f + ar + cp + cr) / 4

    print(f"\n[bold]Overall RAGAS scores (averaged across all questions):[/bold]")
    print(f"  Faithfulness      : {f:.3f}   ← hallucination rate (1.0 = no hallucination)")
    print(f"  Answer relevancy  : {ar:.3f}   ← answers the question (1.0 = perfectly on-topic)")
    print(f"  Context precision : {cp:.3f}   ← retrieval noise     (1.0 = all chunks useful)")
    print(f"  Context recall    : {cr:.3f}   ← completeness        (1.0 = nothing missing)")
    print(f"\n  [bold]Overall average   : {avg_all:.3f}[/bold]")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"\n[bold]Diagnostics:[/bold]")
    checks = [
        (f,  "Faithfulness",      "tighten system prompt — 'use ONLY the context'"),
        (ar, "Answer relevancy",  "prompt Claude to answer exactly what was asked"),
        (cp, "Context precision", "raise reranker threshold or improve chunking"),
        (cr, "Context recall",    "increase n_wide retrieval or expand corpus"),
    ]
    for score, name, fix in checks:
        if score >= 0.8:
            print(f"  [green]{name}: {score:.2f} — good[/green]")
        else:
            print(f"  [yellow]{name}: {score:.2f} — below threshold[/yellow]")
            print(f"  [dim]  Fix: {fix}[/dim]")

    print(f"\n[bold]Phase progression:[/bold]")
    print(f"  Phase 1 baseline  (keyword hit rate) : 6/6 = 100%")
    print(f"  Day 6 RAGAS score (answer quality)   : {avg_all:.3f} / 1.0")
    print(f"\n[dim]Next: Day 7 — hybrid retrieval. Combine Naive + HyDE, rerank combined pool.[/dim]")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if collection.count() == 0:
        print("[red]Collection empty. Run ingest.py first.[/red]")
        sys.exit(1)

    print(f"[bold]Collection: {collection.count()} chunks[/bold]")
    rows   = run_pipeline()
    result = run_ragas_043(rows)
    display_results(result, rows)