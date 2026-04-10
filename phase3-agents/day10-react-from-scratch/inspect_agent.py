import os
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import anthropic
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from rich.console import Console
from tools import run_tool
from agent import parse_llm_output, SYSTEM_PROMPT

load_dotenv(dotenv_path="../../.env")
console = Console()
client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def run_with_trace(question: str):
    print(f"\n[bold]Question:[/bold] {question}\n")

    messages   = [{"role": "user", "content": question}]
    trace      = []
    tools_used = 0
    steps      = 0

    while steps < 8:
        steps += 1
        response   = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        llm_output = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": llm_output})
        parsed = parse_llm_output(llm_output)

        if parsed["type"] == "tool":
            observation = run_tool(parsed["tool_name"], parsed["tool_input"])
            tools_used += 1
            trace.append({
                "step":  steps,
                "type":  "TOOL",
                "tool":  parsed["tool_name"],
                "input": parsed["tool_input"][:60],
                "output": observation[:100],
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

        elif parsed["type"] == "final":
            if tools_used == 0:
                trace.append({
                    "step": steps, "type": "REJECTED",
                    "tool": "—", "input": "—",
                    "output": "Final Answer before any tools — rejected",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "You provided a Final Answer without using any tools. "
                        "This is not allowed. Start with:\n"
                        "Thought: I need to use the dictionary tool first.\n"
                        "Action: dictionary\nAction Input: attention"
                    )
                })
                continue
            trace.append({
                "step":  steps,
                "type":  "FINAL",
                "tool":  "—",
                "input": "—",
                "output": parsed["final_answer"][:100],
            })
            break

        else:
            trace.append({
                "step":  steps,
                "type":  "PARSE ERROR",
                "tool":  "—",
                "input": llm_output[:60],
                "output": "Format not recognised",
            })
            messages.append({
                "role": "user",
                "content": (
                    "Follow this exact format:\n"
                    "Thought: <reasoning>\nAction: <tool>\nAction Input: <input>"
                )
            })

    t = Table(show_header=True, header_style="bold", show_lines=True)
    t.add_column("Step",   width=6)
    t.add_column("Type",   width=12)
    t.add_column("Tool",   width=12)
    t.add_column("Input",  width=28)
    t.add_column("Output", width=36)

    for row in trace:
        color = {
            "TOOL":        "yellow",
            "FINAL":       "green",
            "REJECTED":    "red",
            "PARSE ERROR": "red",
        }.get(row["type"], "white")

        t.add_row(
            str(row["step"]),
            f"[{color}]{row['type']}[/{color}]",
            row["tool"],
            row["input"],
            row["output"],
        )

    console.print(t)
    print(f"\n[bold]Tools used   :[/bold] {tools_used}")
    print(f"[bold]Steps used   :[/bold] {steps}")
    print(f"[bold]LLM calls    :[/bold] {steps + 1}")

if __name__ == "__main__":
    run_with_trace(
        "Use the dictionary to look up 'attention'. "
        "Then use rag_search to find how many attention heads "
        "the base Transformer model uses. "
        "Then use the calculator to compute 8 multiplied by 64."
    )