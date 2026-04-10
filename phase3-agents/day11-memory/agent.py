# phase3-agents/day11-memory/agent.py
import re
import anthropic
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Reuse Day 10 tools
import sys
sys.path.append(str(Path(__file__).parent.parent / "day10-react-from-scratch"))
from tools import calculator, dictionary, rag_search

from memory import Memory

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

console = Console()
client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

memory = Memory()


# ── Memory tools ──────────────────────────────────────────────────────────────

def memory_store(key_value: str) -> str:
    """
    Input format: "key|value|confidence"
    confidence is optional, defaults to 0.9
    Example: "chunking_strategy|semantic chunking splits on meaning|0.95"
    """
    parts = [p.strip() for p in key_value.split("|")]
    if len(parts) < 2:
        return "ERROR: format must be key|value or key|value|confidence"
    key = parts[0]
    value = parts[1]
    confidence = float(parts[2]) if len(parts) >= 3 else 0.9
    memory.store(key, value, confidence=confidence, source="agent_tool")
    return f"Stored: '{key}' = '{value}' [conf={confidence}]"


def memory_retrieve(key: str) -> str:
    key = key.strip()
    result = memory.retrieve(key)
    if result is None:
        return f"No memory found for key: '{key}'"
    tier = result.get("tier", "?")
    conf = result.get("confidence", "?")
    return f"[{tier}-term, conf={conf}] {result['value']}"


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = {
    "calculator": calculator,
    "dictionary": dictionary,
    "rag_search": rag_search,
    "memory_store": memory_store,
    "memory_retrieve": memory_retrieve,
}

TOOL_NAMES = ", ".join(TOOLS.keys())


# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    mem_context = memory.context_block()
    return f"""You are a ReAct agent with memory. You reason step-by-step and use tools.

## Memory context (loaded at session start)
{mem_context}

## Tools available: {TOOL_NAMES}

- calculator(expression)   — safe math eval
- dictionary(word)         — AI/ML term definitions
- rag_search(query)        — search the knowledge base
- memory_store(key|value|confidence) — save a fact for future sessions
- memory_retrieve(key)     — recall a previously stored fact

## Format — use EXACTLY this structure every step:

Thought: <your reasoning>
Action: <tool name>
Action Input: <input to tool>

After receiving Observation, continue with next Thought/Action/Action Input.
When you have enough information, output:

Final Answer: <your complete answer>

Rules:
- Never invent Observations
- Use memory_store to persist any important fact you learn
- Use memory_retrieve before rag_search if you think you've seen something before
- Flag low-confidence memories with ⚠ in your reasoning
- Always use at least one tool before Final Answer
"""


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_response(text: str):
    if "Final Answer:" in text:
        answer = text.split("Final Answer:")[-1].strip()
        return ("final", answer)

    action_match = re.search(r"Action:\s*(.+)", text)
    input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)

    if action_match and input_match:
        tool = action_match.group(1).strip()
        tool_input = input_match.group(1).strip()
        return ("action", tool, tool_input)

    return ("unknown", text)


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(question: str, max_steps: int = 8):
    console.print(Panel(f"[bold cyan]Question:[/] {question}", expand=False))

    messages = [{"role": "user", "content": question}]
    system = build_system_prompt()

    tools_used = []
    step = 0

    while step < max_steps:
        step += 1
        console.print(f"\n[dim]── Step {step} ──[/dim]")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        text = response.content[0].text
        console.print(f"[white]{text}[/white]")

        parsed = parse_response(text)

        # ── Final Answer ──
        if parsed[0] == "final":
            if not tools_used:
                # Guard: must use at least one tool
                nudge = (
                    "You must use at least one tool before giving a Final Answer. "
                    f"Available tools: {TOOL_NAMES}"
                )
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": nudge})
                continue

            console.print(Panel(
                f"[bold green]Final Answer:[/]\n{parsed[1]}",
                title="✓ Complete", expand=False
            ))
            console.print(f"\n[dim]Steps: {step} | Tools used: {tools_used}[/dim]")
            return parsed[1]

        # ── Tool call ──
        elif parsed[0] == "action":
            _, tool_name, tool_input = parsed
            tool_name_clean = tool_name.lower().strip()

            if tool_name_clean not in TOOLS:
                obs = (
                    f"ERROR: '{tool_name}' is not a valid tool. "
                    f"Valid tools: {TOOL_NAMES}"
                )
            else:
                console.print(f"[yellow]→ Calling {tool_name_clean}({tool_input!r})[/yellow]")
                try:
                    obs = TOOLS[tool_name_clean](tool_input)
                except Exception as e:
                    obs = f"Tool error: {e}"
                tools_used.append(tool_name_clean)

            console.print(f"[green]Observation: {obs}[/green]")

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        # ── Parse failure ──
        else:
            nudge = (
                f"Format error. Use exactly:\n"
                f"Thought: ...\nAction: <one of {TOOL_NAMES}>\nAction Input: ...\n"
                f"or: Final Answer: ..."
            )
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": nudge})

    return "Agent reached max steps without a Final Answer."


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        # Tests memory_store then memory_retrieve across two calls
        "What is semantic chunking? Store the definition in memory with key 'semantic_chunking'.",
        "Retrieve the memory for 'semantic_chunking' and tell me what it says.",
    ]
    for q in questions:
        run_agent(q)
        print("\n" + "="*60 + "\n")