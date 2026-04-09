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
from hybrid import hybrid_rag

load_dotenv(dotenv_path="../../.env")
console = Console()

# ── Eval set — same as Day 6 for direct comparison ───────────────────────────
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
    print(f"\n[bold cyan]Running hybrid pipeline on {len(EVAL_SET)} questions...[/bold cyan]\n")
    rows = []
    for item in EVAL_SET:
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"[dim]Processing:[/dim] {q[:60]}...")
        result = hybrid_rag(q)
        rows.append({
            "question":     q,
            "answer":       result["answer"],
            "contexts":     [c["chunk"] for c in result["top_chunks"]],
            "ground_truth": gt,
            "merged_count": result["merged"],
        })
        print(f"  [dim]Pool: {result['merged']} candidates → {len(result['top_chunks'])} kept[/dim]")
    return rows

def run_ragas(rows):
    print(f"\n[bold cyan]Running RAGAS 0.4.3 evaluation...[/bold cyan]")
    print("[dim]Takes 2-3 minutes — making LLM calls to score each metric.[/dim]\n")

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
    console.print(Rule("[bold]Day 7 — Hybrid Retrieval RAGAS Results[/bold]"))

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

        for _, row in df.iterrows():
            fq  = safe_float(row, "faithfulness")
            arq = safe_float(row, "answer_relevancy")
            cpq = safe_float(row, "context_precision")
            crq = safe_float(row, "context_recall")
            avg = (fq + arq + cpq + crq) / 4
            verdict = "[green]Strong[/green]"       if avg >= 0.75 else \
                      "[yellow]Needs work[/yellow]" if avg >= 0.55 else \
                      "[red]Failing[/red]"
            t.add_row(
                str(row.get("question", ""))[:36],
                fmt(fq), fmt(arq), fmt(cpq), fmt(crq), verdict
            )
        console.print(t)

    except Exception as e:
        print(f"[yellow]Per-question table skipped: {e}[/yellow]\n")

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

    # ── Comparison table: Day 6 vs Day 7 ─────────────────────────────────────
    print(f"\n[bold]Day 6 vs Day 7 — direct comparison:[/bold]")

    c = Table(show_header=True, header_style="bold")
    c.add_column("Metric",             width=20)
    c.add_column("Day 6 (reranked)",   width=18)
    c.add_column("Day 7 (hybrid)",     width=18)
    c.add_column("Delta",              width=12)

    day6 = {
        "faithfulness":      0.896,
        "answer_relevancy":  0.741,
        "context_precision": 0.667,
        "context_recall":    0.250,
        "overall":           0.638,
    }
    day7 = {
        "faithfulness":      f,
        "answer_relevancy":  ar,
        "context_precision": cp,
        "context_recall":    cr,
        "overall":           avg_all,
    }

    for metric, d6, d7 in [
        ("Faithfulness",      day6["faithfulness"],      day7["faithfulness"]),
        ("Answer relevancy",  day6["answer_relevancy"],  day7["answer_relevancy"]),
        ("Context precision", day6["context_precision"], day7["context_precision"]),
        ("Context recall",    day6["context_recall"],    day7["context_recall"]),
        ("Overall average",   day6["overall"],           day7["overall"]),
    ]:
        delta = d7 - d6
        delta_str = f"[green]+{delta:.3f}[/green]" if delta > 0.01 else \
                    f"[red]{delta:.3f}[/red]"       if delta < -0.01 else \
                    "[dim]~0[/dim]"
        c.add_row(metric, f"{d6:.3f}", f"{d7:.3f}", delta_str)

    console.print(c)

    print(f"\n[bold]Overall average   : {avg_all:.3f}[/bold]")
    print(f"[bold]Day 6 baseline    : 0.638[/bold]")

    delta_overall = avg_all - 0.638
    if delta_overall > 0.02:
        print(f"[green]Hybrid retrieval improved overall score by {delta_overall:+.3f}[/green]")
    elif delta_overall < -0.02:
        print(f"[yellow]Score dropped {delta_overall:.3f} — check corpus and ground truths[/yellow]")
    else:
        print(f"[dim]Score held steady — hybrid retrieval maintained quality[/dim]")

    print(f"\n[bold]Phase progression:[/bold]")
    print(f"  Phase 1 naive RAG  : 6/6 keyword hit = 100%")
    print(f"  Day 6 RAGAS        : 0.638")
    print(f"  Day 7 hybrid RAGAS : {avg_all:.3f}")
    print(f"\n[dim]Next: Day 8 — query expansion.[/dim]")

if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    col = chroma_client.get_or_create_collection(
        name="day7", metadata={"hnsw:space": "cosine"}
    )
    if col.count() == 0:
        print("[red]Collection empty. Run ingest.py first.[/red]")
        sys.exit(1)
    print(f"[bold]Collection: {col.count()} chunks[/bold]")
    rows   = run_pipeline()
    result = run_ragas(rows)
    display_results(result, rows)