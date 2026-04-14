# phase3-agents/day13-capstone/orchestrator.py
import sys
from pathlib import Path
from rich.console import Console

# Pull in Day 12 agents
sys.path.append(str(Path(__file__).parent.parent / "day12-multi-agent"))
from agents import RAGAgent, MathAgent, MemoryAgent

# Pull in Day 11 memory
sys.path.append(str(Path(__file__).parent.parent / "day11-memory"))
from memory import Memory

from planner import plan

console = Console()

AGENT_MAP = {
    "rag":    RAGAgent(),
    "math":   MathAgent(),
    "memory": MemoryAgent(),
}

memory = Memory()

def run_sub_task(sub_task: dict, context: list[dict]) -> dict:
    agent_name = sub_task["agent"]
    task = sub_task["task"]

    if context:
        context_lines = "\n".join(
            f"Sub-task {r['id']} ({r['agent']}): {r['result']}"
            for r in context
            if "reached max steps" not in r["result"]  # ← skip failed results
        )
        enriched_task = (
            f"{task}\n\nContext from earlier sub-tasks:\n{context_lines}"
            if context_lines else task
        )
    else:
        enriched_task = task

    if agent_name not in AGENT_MAP:
        console.print(f"[red]Unknown agent '{agent_name}', skipping sub-task {sub_task['id']}[/red]")
        return {**sub_task, "result": f"ERROR: unknown agent '{agent_name}'"}

    console.print(f"\n[bold]Sub-task {sub_task['id']} → {agent_name} agent[/bold]")
    console.print(f"[dim]Task: {task}[/dim]")

    agent = AGENT_MAP[agent_name]
    result = agent.run(enriched_task)

    # Guard — catch max steps error before it propagates
    if "reached max steps" in result:
        console.print(f"[red]Sub-task {sub_task['id']} failed: {result}[/red]")
        result = f"Sub-task {sub_task['id']} could not be completed."

    return {**sub_task, "result": result}


def orchestrate(question: str) -> list[dict]:
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
        result = run_sub_task(sub_task, context=results)
        results.append(result)

        # Persist any facts the memory agent produced
        if sub_task["agent"] == "memory":
            memory.store(
                key=f"day13_subtask_{sub_task['id']}",
                value=result["result"],
                confidence=0.9,
                source="orchestrator"
            )

    return results