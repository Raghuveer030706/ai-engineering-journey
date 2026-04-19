# phase4-mcp/day16-capstone/run.py
import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

sys.path.append(str(Path(__file__).parent.parent / "day15-multiagent-mcp"))

from client.mcp_client import MCPClient
from orchestrator import orchestrate
from synthesizer import synthesize

console = Console()

questions = [
    # Single agent — rag
    "What is cross-encoder reranking and when does it help?",

    # Two agents — rag + math
    "What was the Phase 2 RAGAS improvement and what is that as a percentage?",

    # Three agents — rag + math + memory
    "What was the Day 9 RAGAS score, multiply it by 100, and store as 'day9_pct'.",

    # Four agents — rag + math + memory + fetch
    "What was the Phase 2 score, calculate the percentage gain from Phase 1, "
    "store it as 'phase2_gain_pct', then fetch https://www.anthropic.com "
    "and summarise what you find.",

    # Memory recall — tests persistence
    "Recall the memory for 'day9_pct' and explain what it represents.",
]


async def main():
    console.print("[bold]Initializing MCP client...[/bold]")
    mcp = MCPClient()
    await mcp.initialize()

    for q in questions:
        console.print(Panel(f"[bold cyan]Question:[/] {q}", expand=False))
        results = await orchestrate(q, mcp)
        synthesize(q, results)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())