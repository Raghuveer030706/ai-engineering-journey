# phase3-agents/day12-multi-agent/supervisor.py
from unittest import result

import anthropic
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from agents import RAGAgent, MathAgent, MemoryAgent

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

console = Console()
client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

AGENTS = {
    "rag":    RAGAgent(),
    "math":   MathAgent(),
    "memory": MemoryAgent(),
}

ROUTER_PROMPT = """You are a supervisor that routes questions to specialist agents.

Agents available:
- rag    : questions about AI/ML concepts, research papers, technical knowledge
- math   : arithmetic, calculations, numeric problems
- memory : storing a fact for later, recalling a previously stored fact

Reply with ONLY the agent name. One word. No explanation.
Options: rag, math, memory"""


def route(question: str) -> str:
    """Ask the LLM which agent should handle this question."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        system=ROUTER_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    agent_name = response.content[0].text.strip().lower()
    if agent_name not in AGENTS:
        console.print(f"[red]Router returned unknown agent '{agent_name}', defaulting to rag[/red]")
        return "rag"
    return agent_name


# supervisor.py — replace the synthesize block inside run()

def synthesize(question: str, agent_name: str, result: str) -> str:
    """Synthesizes specialist result into clean user-facing answer."""

    system = (
        "You are a result presenter. A specialist agent has already completed the task "
        "and returned a verified result. Your only job is to present that result clearly "
        "to the user. Do not question the result. Do not say you are unable to help. "
        "Do not add caveats. Trust the agent result completely and relay it cleanly."
    )

    prompt = (
        f"User question: {question}\n\n"
        f"The {agent_name} agent returned this result:\n{result}\n\n"
        f"Present this result as a clean final answer to the user."
    )

    final = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return final.content[0].text.strip()


def run(question: str) -> str:
    console.print(Panel(f"[bold cyan]Question:[/] {question}", expand=False))

    # Step 1: route
    agent_name = route(question)
    console.print(f"[bold]Router → {agent_name} agent[/bold]")

    # Step 2: delegate
    specialist = AGENTS[agent_name]
    result = specialist.run(question)

    # Temporary debug line
    console.print(f"[red]DEBUG — agent result:[/] {result}")

    # Step 3: synthesize
    answer = synthesize(question, agent_name, result)

    console.print(Panel(
        f"[bold green]Final Answer:[/]\n{answer}",
        title="✓ Complete", expand=False
    ))
    return answer