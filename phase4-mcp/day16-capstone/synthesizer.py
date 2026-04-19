# phase4-mcp/day16-capstone/synthesizer.py
import anthropic
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"
console = Console()

SYNTHESIZER_SYSTEM = (
    "You are a result presenter. Specialist agents have already completed "
    "each part of the task and returned verified results. "
    "Combine those results into one clean coherent answer. "
    "Do not question the results. Do not say you are unable to help. "
    "Do not mention internal agents or sub-tasks. "
    "Trust the agent results completely."
)


def synthesize(question: str, results: list[dict]) -> str:
    # Only include successful results
    successful = [
        r for r in results
        if "could not be completed" not in r["result"]
        and not r["result"].startswith("ERROR")
    ]

    if not successful:
        return "All sub-tasks failed. No answer could be produced."

    results_block = "\n\n".join(
        f"Part {r['id']} ({r['agent']} agent):\n{r['result']}"
        for r in successful
    )

    prompt = (
        f"User question: {question}\n\n"
        f"Agent results:\n{results_block}\n\n"
        f"Write the final answer."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYNTHESIZER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.content[0].text.strip()
    console.print(Panel(
        f"[bold green]Final Answer:[/]\n{answer}",
        title="✓ Phase 4 Capstone Complete",
        expand=False
    ))
    return answer