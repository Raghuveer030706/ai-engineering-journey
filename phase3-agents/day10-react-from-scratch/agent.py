import os
import re
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import anthropic
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from tools import describe_tools, run_tool

load_dotenv(dotenv_path="../../.env")
client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
console = Console()

SYSTEM_PROMPT = f"""You are a research assistant. You must use tools to answer every question.

TOOLS AVAILABLE:
{describe_tools()}

MANDATORY FORMAT — you must follow this exactly, every single response:

Thought: <your reasoning>
Action: <exact tool name from the list above>
Action Input: <input to the tool, plain text, no quotes>

After you receive an Observation, output the next step in the same format.

After you have used a tool for EVERY part of the question, you may conclude:

Thought: <summary of what tools returned>
Final Answer: <your answer based only on what the tools returned>

STRICT RULES:
1. Your very first response MUST start with Thought: then Action: — never Final Answer first
2. You must call one tool per step — never skip a tool
3. Never answer from memory — always use a tool
4. Action Input must be plain text — no quotes, no brackets
5. Do not write Observation: yourself — that comes from the system"""

def parse_llm_output(text: str) -> dict:
    """
    Parses LLM output and returns a dict with keys:
      type: "tool" | "final" | "unknown"
      tool_name: str or None
      tool_input: str or None
      final_answer: str or None
    """
    # Check Final Answer
    final_match = re.search(
        r"Final Answer\s*:\s*(.+)",
        text, re.DOTALL | re.IGNORECASE
    )

    # Check Action
    action_match = re.search(
        r"Action\s*:\s*(\w+)",
        text, re.IGNORECASE
    )
    input_match = re.search(
        r"Action Input\s*:\s*(.+?)(?=\nObservation|\nThought|\nAction|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )

    if action_match and input_match:
        tool_name  = action_match.group(1).strip().lower()
        tool_input = input_match.group(1).strip()
        # Strip surrounding quotes if present
        tool_input = tool_input.strip('"\'')
        return {"type": "tool", "tool_name": tool_name,
                "tool_input": tool_input, "final_answer": None}

    if final_match:
        return {"type": "final", "tool_name": None,
                "tool_input": None,
                "final_answer": final_match.group(1).strip()}

    return {"type": "unknown", "tool_name": None,
            "tool_input": None, "final_answer": None}

def run_agent(question: str, max_steps: int = 8) -> str:
    console.print(Rule(f"[bold]{question[:70]}[/bold]"))

    messages   = [{"role": "user", "content": question}]
    tools_used = 0
    steps      = 0

    while steps < max_steps:
        steps += 1
        console.print(f"\n[dim]── Step {steps} ──[/dim]")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        llm_output = response.content[0].text.strip()
        console.print(f"[cyan]{llm_output}[/cyan]")
        messages.append({"role": "assistant", "content": llm_output})

        parsed = parse_llm_output(llm_output)

        # ── Tool call ──────────────────────────────────────────────────────
        if parsed["type"] == "tool":
            tool_name  = parsed["tool_name"]
            tool_input = parsed["tool_input"]
            console.print(f"\n[bold yellow]Tool:[/bold yellow] {tool_name}({tool_input[:80]})")
            observation = run_tool(tool_name, tool_input)
            console.print(f"[bold yellow]Result:[/bold yellow] {observation[:200]}")
            tools_used += 1
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        # ── Final Answer ───────────────────────────────────────────────────
        elif parsed["type"] == "final":
            if tools_used == 0:
                # Reject — no tools were called yet
                console.print(
                    "[red]Rejected Final Answer — no tools used yet.[/red]"
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "You provided a Final Answer without using any tools. "
                        "This is not allowed. You must use the dictionary, "
                        "rag_search, and calculator tools as instructed. "
                        "Start with: Thought: I need to use the dictionary tool first.\n"
                        "Action: dictionary\nAction Input: attention"
                    )
                })
                continue

            final = parsed["final_answer"]
            console.print(f"\n[bold green]Final Answer:[/bold green] {final}")
            return final

        # ── Unknown format ─────────────────────────────────────────────────
        else:
            console.print("[red]Format not recognised — nudging.[/red]")
            messages.append({
                "role": "user",
                "content": (
                    "Your response did not follow the required format. "
                    "You must respond with:\n"
                    "Thought: <your reasoning>\n"
                    "Action: <tool name>\n"
                    "Action Input: <input>\n\n"
                    "Available tools: dictionary, rag_search, calculator"
                )
            })

    # Max steps — force final answer
    console.print(f"[yellow]Max steps reached. Requesting final answer.[/yellow]")
    messages.append({
        "role": "user",
        "content": "Maximum steps reached. Provide your Final Answer now."
    })
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    final = response.content[0].text
    console.print(f"\n[bold green]Final Answer (forced):[/bold green] {final}")
    return final

if __name__ == "__main__":
    TEST_QUESTIONS = [
        (
            "Use the dictionary tool to look up 'transformer'. "
            "Then use rag_search to find the BLEU score the Transformer "
            "achieved on English-to-German translation. "
            "Then use the calculator to compute 512 divided by 64."
        ),
        (
            "Use rag_search to find what optimizer was used to train "
            "the Transformer model. Then use the calculator to compute "
            "0.9 multiplied by 0.98."
        ),
    ]
    for q in TEST_QUESTIONS:
        run_agent(q)
        print()