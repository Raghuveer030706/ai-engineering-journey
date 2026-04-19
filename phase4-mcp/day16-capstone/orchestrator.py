# phase4-mcp/day16-capstone/orchestrator.py
import sys
import asyncio
from pathlib import Path
from rich.console import Console

# Pull in Day 15 agents and MCP client
sys.path.append(str(Path(__file__).parent.parent / "day15-multiagent-mcp"))

from agents import RAGAgent, MathAgent, MemoryAgent, FetchAgent
from client.mcp_client import MCPClient
from planner import plan

console = Console()

VALID_AGENTS = ["rag", "math", "memory", "fetch"]


def build_agent_map(mcp: MCPClient) -> dict:
    """Create fresh agent instances with shared MCP client."""
    return {
        "rag":    RAGAgent(mcp),
        "math":   MathAgent(mcp),
        "memory": MemoryAgent(mcp),
        "fetch":  FetchAgent(mcp),
    }


async def run_sub_task(sub_task: dict, context: list[dict], mcp: MCPClient) -> dict:
    """
    Runs a single sub-task.
    Injects successful prior results as context.
    Filters failed results so error strings don't poison downstream reasoning.
    """
    agent_name = sub_task["agent"]
    task = sub_task["task"]

    # Inject prior successful results
    successful = [r for r in context if "reached max steps" not in r["result"]
                  and not r["result"].startswith("ERROR")]
    if successful:
        context_lines = "\n".join(
            f"Sub-task {r['id']} ({r['agent']}): {r['result']}"
            for r in successful
        )
        enriched_task = f"{task}\n\nContext from earlier sub-tasks:\n{context_lines}"
    else:
        enriched_task = task

    if agent_name not in VALID_AGENTS:
        console.print(f"[red]Unknown agent '{agent_name}', skipping[/red]")
        return {**sub_task, "result": f"ERROR: unknown agent '{agent_name}'"}

    console.print(f"\n[bold]Sub-task {sub_task['id']} → {agent_name} agent[/bold]")
    console.print(f"[dim]Task: {task}[/dim]")

    agent_map = build_agent_map(mcp)
    agent = agent_map[agent_name]
    result = await agent.run(enriched_task)

    # Guard — catch max steps before it propagates
    if "reached max steps" in result:
        console.print(f"[red]Sub-task {sub_task['id']} failed: {result}[/red]")
        result = f"Sub-task {sub_task['id']} could not be completed."

    return {**sub_task, "result": result}


async def orchestrate(question: str, mcp: MCPClient) -> list[dict]:
    """
    Full pipeline: plan → run each sub-task in order → return all results.
    """
    console.print(f"\n[bold cyan]Planner decomposing:[/] {question}")

    sub_tasks = plan(question)

    console.print(f"[dim]Plan: {len(sub_tasks)} sub-task(s)[/dim]")
    for st in sub_tasks:
        console.print(f"  {st['id']}. [{st['agent']}] {st['task']}")

    results = []
    for sub_task in sub_tasks:
        result = await run_sub_task(sub_task, context=results, mcp=mcp)
        results.append(result)

    return results