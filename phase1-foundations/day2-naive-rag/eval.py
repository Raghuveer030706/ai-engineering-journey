import os
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv(dotenv_path="../../.env")

from rag import rag
from rich import print
from rich.table import Table

# Known question → keyword that MUST appear in retrieved chunks
# This tests retrieval quality, not LLM quality
EVAL_SET = [
    {"question": "What is RAG?",                              "expected_keyword": "Retrieval-Augmented"},
    {"question": "How do vector databases work?",             "expected_keyword": "embeddings"},
    {"question": "What is cosine similarity?",                "expected_keyword": "cosine"},
    {"question": "How do agents decide what to do?",          "expected_keyword": "tools"},
    {"question": "What is the attention mechanism?",          "expected_keyword": "attention"},
    {"question": "What does fine-tuning do?",                 "expected_keyword": "pre-trained"},
    {"question": "What is MCP?",                              "expected_keyword": "Model Context Protocol"},
    {"question": "What are embeddings?",                      "expected_keyword": "numerical"},
]

print("[bold]--- RAG Eval Harness ---[/bold]\n")
print(f"Running {len(EVAL_SET)} eval questions...\n")

table = Table(show_header=True, header_style="bold")
table.add_column("Question",          width=38)
table.add_column("Expected keyword",  width=22)
table.add_column("Pass", width=6)
table.add_column("Top chunk (truncated)")

passed = 0

for item in EVAL_SET:
    result   = rag(item["question"], n_results=3)
    all_text = " ".join(result["chunks"]).lower()
    keyword  = item["expected_keyword"].lower()
    hit      = keyword in all_text

    if hit:
        passed += 1

    table.add_row(
        item["question"][:38],
        item["expected_keyword"],
        "[green]YES[/green]" if hit else "[red]NO[/red]",
        result["chunks"][0][:55],
    )

print(table)

score = passed / len(EVAL_SET) * 100
print(f"\n[bold]Retrieval hit rate: {passed}/{len(EVAL_SET)} = {score:.0f}%[/bold]")

if score == 100:
    print("[green]Perfect retrieval. Baseline locked in.[/green]")
elif score >= 75:
    print("[yellow]Good baseline. Phase 2 techniques will push this higher.[/yellow]")
else:
    print("[red]Retrieval needs improvement. Check your chunks and embedding model.[/red]")

print("\nThis score is your Day 2 baseline.")
print("Save it -- in Phase 2 you will compare advanced RAG against this number.")