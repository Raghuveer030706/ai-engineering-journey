import os
import sys
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from rich.console import Console
from rich.rule import Rule
from pipeline import phase2_rag

load_dotenv(dotenv_path="../../.env")
console = Console()

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
        "ground_truth": "The Adam optimizer was used with β1 = 0.9, β2 = 0.98 and ϵ = 10−9. The learning rate was varied during training using a schedule that increases linearly for warmup steps then decreases proportionally to the inverse square root of the step number."
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

def run_pipeline():
    print(f"\n[bold cyan]Running Phase 2 capstone pipeline — {len(EVAL_SET)} questions...[/bold cyan]\n")
    rows = []
    for item in EVAL_SET:
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"[dim]Processing:[/dim] {q[:60]}...")
        result = phase2_rag(q)
        exp_note = "expanded" if result["expansion_used"] else "hybrid only"
        print(f"  [dim]{result['merged']} candidates → "
              f"{len(result['top_chunks'])} kept ({exp_note})[/dim]")
        rows.append({
            "question":        q,
            "answer":          result["answer"],
            "contexts":        [c["chunk"] for c in result["top_chunks"]],
            "ground_truth":    gt,
            "expansion_used":  result["expansion_used"],
        })
    return rows

def run_ragas(rows):
    print(f"\n[bold cyan]Running RAGAS 0.4.3 evaluation...[/bold cyan]")
    print("[dim]Takes 2-3 minutes.[/dim]\n")

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

    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in rows],
        "answer":       [r["answer"]       for r in rows],
        "contexts":     [r["contexts"]     for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    llm = LangchainLLMWrapper(ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2048,
        temperature=0,
    ))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ))

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = emb

    return evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)

def display_results(result, rows):
    console.print(Rule("[bold]Day 9 — Phase 2 Capstone RAGAS Results[/bold]"))

    # ── Per-question table ────────────────────────────────────────────────────
    try:
        df = result.to_pandas()
        df.columns = [str(c).lower().strip() for c in df.columns]
        if "question" not in df.columns:
            df.insert(0, "question", [r["question"] for r in rows])

        t = Table(show_header=True, header_style="bold", show_lines=True)
        t.add_column("Question",         width=36)
        t.add_column("Faithfulness",     width=13)
        t.add_column("Ans relevancy",    width=13)
        t.add_column("Ctx precision",    width=13)
        t.add_column("Ctx recall",       width=11)
        t.add_column("Expansion",        width=10)
        t.add_column("Verdict",          width=14)

        def fmt(v):
            try:
                v = float(v) if v is not None else 0.0
                if v >= 0.8:   return f"[green]{v:.2f}[/green]"
                elif v >= 0.6: return f"[yellow]{v:.2f}[/yellow]"
                else:          return f"[red]{v:.2f}[/red]"
            except Exception:
                return "[dim]n/a[/dim]"

        def safe_float(row, col):
            try:
                v = row.get(col, 0)
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

        for i, (_, row) in enumerate(df.iterrows()):
            fq  = safe_float(row, "faithfulness")
            arq = safe_float(row, "answer_relevancy")
            cpq = safe_float(row, "context_precision")
            crq = safe_float(row, "context_recall")
            avg = (fq + arq + cpq + crq) / 4
            verdict = "[green]Strong[/green]"       if avg >= 0.75 else \
                      "[yellow]Needs work[/yellow]" if avg >= 0.55 else \
                      "[red]Failing[/red]"
            exp_used = rows[i]["expansion_used"] if i < len(rows) else False
            exp_str  = "[yellow]Yes[/yellow]" if exp_used else "[dim]No[/dim]"
            t.add_row(
                str(row.get("question", ""))[:36],
                fmt(fq), fmt(arq), fmt(cpq), fmt(crq),
                exp_str, verdict
            )
        console.print(t)

    except Exception as e:
        print(f"[yellow]Per-question table skipped: {e}[/yellow]\n")

    # ── Compute overall scores from DataFrame ─────────────────────────────────
    def col_avg(col):
        try:
            df2 = result.to_pandas()
            df2.columns = [str(c).lower().strip() for c in df2.columns]
            if col in df2.columns:
                vals = df2[col].dropna().astype(float)
                return float(vals.mean()) if len(vals) > 0 else 0.0
            return 0.0
        except Exception:
            return 0.0

    f  = col_avg("faithfulness")
    ar = col_avg("answer_relevancy")
    cp = col_avg("context_precision")
    cr = col_avg("context_recall")
    avg_all = (f + ar + cp + cr) / 4

    # ── Full phase comparison table ───────────────────────────────────────────
    print(f"\n[bold]Complete Phase 2 progression:[/bold]")

    c = Table(show_header=True, header_style="bold", show_lines=True)
    c.add_column("Metric",              width=20)
    c.add_column("Phase 1 naive",       width=14)
    c.add_column("Day 6 reranked",      width=14)
    c.add_column("Day 7 hybrid",        width=14)
    c.add_column("Day 8 expanded",      width=14)
    c.add_column("Day 9 capstone",      width=14)
    c.add_column("Total gain",          width=12)

    history = {
        "Faithfulness":      (None, 0.896, 0.883, 0.755, f),
        "Answer relevancy":  (None, 0.741, 0.873, 0.772, ar),
        "Context precision": (None, 0.667, 0.723, 0.531, cp),
        "Context recall":    (None, 0.250, 0.750, 0.750, cr),
        "Overall average":   (None, 0.638, 0.807, 0.702, avg_all),
    }

    for metric, (p1, d6, d7, d8, d9) in history.items():
        total = d9 - d6
        gain_str = f"[green]+{total:.3f}[/green]" if total > 0.01 else \
                   f"[red]{total:.3f}[/red]"       if total < -0.01 else \
                   "[dim]~0[/dim]"
        c.add_row(
            metric,
            "[dim]no RAGAS[/dim]",
            f"{d6:.3f}",
            f"{d7:.3f}",
            f"{d8:.3f}",
            f"{d9:.3f}",
            gain_str,
        )
    console.print(c)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[bold]Phase 2 capstone score  : {avg_all:.3f}[/bold]")
    print(f"[bold]Phase 1 naive baseline  : 0.638[/bold]")

    total_gain = avg_all - 0.638
    if total_gain > 0.02:
        print(f"\n[green]Phase 2 improved overall RAGAS by {total_gain:+.3f} "
              f"over Phase 1 naive RAG.[/green]")

    # ── What each technique contributed ───────────────────────────────────────
    print(f"\n[bold]What each technique contributed:[/bold]")
    print(f"  Day 4 HyDE          → better retrieval on vague queries")
    print(f"  Day 5 Reranking     → rank movement, correct chunks surfaced")
    print(f"  Day 6 RAGAS         → honest measurement, revealed recall gap at 0.25")
    print(f"  Day 7 Hybrid        → recall 0.25 → 0.75, overall 0.638 → 0.807")
    print(f"  Day 8 Expansion     → recall maintained, precision tradeoff identified")
    print(f"  Day 9 Capstone      → selective expansion preserves Day 7 precision")

    print(f"\n[bold]Selective expansion triggered on:[/bold]")
    triggered = [r["question"][:50] for r in rows if r["expansion_used"]]
    skipped   = [r["question"][:50] for r in rows if not r["expansion_used"]]
    for q in triggered:
        print(f"  [yellow]expanded:[/yellow] {q}")
    for q in skipped:
        print(f"  [dim]skipped :[/dim]  {q}")

    print(f"\n[dim]Phase 2 complete. Next: Phase 3 — Agents.[/dim]")

if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    col = chroma_client.get_or_create_collection(
        name="day9", metadata={"hnsw:space": "cosine"}
    )
    if col.count() == 0:
        print("[red]Collection empty. Run ingest.py first.[/red]")
        sys.exit(1)
    print(f"[bold]Collection: {col.count()} chunks[/bold]")
    rows   = run_pipeline()
    result = run_ragas(rows)
    display_results(result, rows)