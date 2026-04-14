# phase3-agents/day13-capstone/synthesizer.py
import anthropic
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"
console = Console()

def synthesize(question: str, results: list[dict]) -> str:
    results_block = "\n\n".join(
        f"Part {r['id']} ({r['agent']} agent):\n{r['result']}"
        for r in results
    )

    prompt = (
        f"User question: {question}\n\n"
        f"Agent results:\n{results_block}\n\n"
        f"Write the final answer."
    )

    system = (
        "You are a result presenter. Specialist agents have already completed "
        "each part of the task and returned verified results. "
        "Your only job is to combine those results into one clean, coherent answer. "
        "Do not question the results. Do not say you are unable to help. "
        "Do not add facts not present in the agent results. "
        "Do not mention internal agents or sub-tasks to the user. "
        "Trust the agent results completely and present them as one unified answer."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.content[0].text.strip()
    console.print(Panel(
        f"[bold green]Final Answer:[/]\n{answer}",
        title="✓ Capstone Complete", expand=False
    ))
    return answer