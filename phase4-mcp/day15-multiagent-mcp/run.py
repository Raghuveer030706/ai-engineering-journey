# phase4-mcp/day15-multiagent-mcp/run.py
import asyncio
from rich.console import Console
from client.mcp_client import MCPClient
from supervisor import run

console = Console()

questions = [
    "What was the Phase 2 RAGAS capstone score?",           # → rag
    "What is 0.827 minus 0.638 multiplied by 100?",         # → math
    "Store this fact: day15_complete|MCP multi-agent done", # → memory
    "Recall the fact stored for day15_complete.",            # → memory
]

async def main():
    console.print("[bold]Initializing MCP client...[/bold]")
    mcp = MCPClient()
    await mcp.initialize()

    for q in questions:
        await run(q, mcp)
        print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())