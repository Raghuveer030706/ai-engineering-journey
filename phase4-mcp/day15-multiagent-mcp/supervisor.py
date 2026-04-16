# phase4-mcp/day15-multiagent-mcp/supervisor.py
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

ROUTER_PROMPT = """You are a supervisor that routes questions to specialist agents.

Agents available:
- rag    : facts, concepts, scores from the knowledge base or project docs
- math   : arithmetic, calculations, numeric problems
- memory : storing a fact for later, recalling a previously stored fact

Reply with ONLY the agent name. One word. No explanation.
Options: rag, math, memory"""

SYNTHESIZER_SYSTEM = (
    "You are a result presenter. A specialist agent has already completed "
    "the task and returned a verified result. Your only job is to present "
    "that result clearly to the user. Do not question the result. "
    "Do not say you are unable to help. Trust the agent result completely."
)


def route(question: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        system=ROUTER_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    agent_name = response.content[0].text.strip().lower()
    if agent_name not in ["rag", "math", "memory"]:
        console.print(
            f"[red]Router returned '{agent_name}', defaulting to rag[/red]"
        )
        return "rag"
    return agent_name


def synthesize(question: str, agent_name: str, result: str) -> str:
    prompt = (
        f"User question: {question}\n\n"
        f"The {agent_name} agent returned:\n{result}\n\n"
        f"Present this as a clean final answer."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYNTHESIZER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


async def run(question: str, mcp) -> str:
    console.print(Panel(f"[bold cyan]Question:[/] {question}", expand=False))

    agent_name = route(question)
    console.print(f"[bold]Router → {agent_name} agent[/bold]")

    agents = {
        "rag":    RAGAgent(mcp),
        "math":   MathAgent(mcp),
        "memory": MemoryAgent(mcp),
    }

    specialist = agents[agent_name]
    result = specialist.run(question)
    result = await result

    answer = synthesize(question, agent_name, result)
    console.print(Panel(
        f"[bold green]Final Answer:[/]\n{answer}",
        title="✓ Complete", expand=False
    ))
    return answer