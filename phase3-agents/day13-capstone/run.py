# phase3-agents/day13-capstone/run.py
from orchestrator import orchestrate
from synthesizer import synthesize
from rich.console import Console
from rich.panel import Panel

console = Console()

# Test questions — each exercises a different decomposition pattern
questions = [
    # Single agent — simple pass-through
    "What is cross-encoder reranking?",

    # Two agents — knowledge + math
    "How many chunks did we end up with after filtering on Day 3, "
    "and what percentage were removed from the original 180?",

    # Three agents — knowledge + math + memory
    "What was the RAGAS score improvement from Phase 1 to Phase 2, "
    "calculate the percentage gain, "
    "and store the result in memory with key 'phase2_gain'.",

    # Memory recall — tests persistence
    "Recall the memory for 'phase2_gain' and explain what it means.",
]

for q in questions:
    console.print(Panel(f"[bold cyan]Question:[/] {q}", expand=False))
    results = orchestrate(q)
    synthesize(q, results)
    print("\n" + "=" * 60 + "\n")