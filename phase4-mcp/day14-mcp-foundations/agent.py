# phase4-mcp/day14-mcp-foundations/agent.py
import json
import asyncio
import re
import anthropic
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from client.mcp_client import MCPClient

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

console = Console()
client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"


def build_system_prompt(tool_descriptions: str) -> str:
    return f"""You are a ReAct agent connected to MCP servers.
You reason step by step and call tools to answer questions.

## Available tools (served via MCP)
{tool_descriptions}

## Format — use EXACTLY this structure every step:

Thought: <your reasoning>
Action: <tool name>
Action Input: {{"key": "value"}}

After receiving an Observation, continue with the next Thought/Action.
When you have enough information:

Final Answer: <your complete answer>

Rules:
- Action Input must be valid JSON matching the tool input schema
- Never invent an Observation — wait for the real one
- Never write Action and Final Answer in the same response
- Always use at least one tool before Final Answer
"""


def parse_response(text: str):
    # Final Answer — only if no Action on same line
    if "Final Answer:" in text:
        lines = text.split("\n")
        has_action = any(l.strip().startswith("Action:") for l in lines)
        if not has_action:
            return ("final", text.split("Final Answer:")[-1].strip())

    action_match = re.search(r"Action:\s*(.+)", text)
    if not action_match:
        return ("unknown", text)

    tool_name = action_match.group(1).strip()

    # Try to extract JSON block — greedy to handle nested braces
    input_match = re.search(r"Action Input:\s*(\{[\s\S]*?\})(?:\n|$)", text)
    if input_match:
        try:
            tool_args = json.loads(input_match.group(1).strip())
            return ("action", tool_name, tool_args)
        except json.JSONDecodeError:
            pass

    # Fallback — raw string after Action Input:
    raw_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
    if raw_match:
        raw = raw_match.group(1).strip()
        try:
            tool_args = json.loads(raw)
            return ("action", tool_name, tool_args)
        except json.JSONDecodeError:
            return ("action_raw", tool_name, raw)

    return ("unknown", text)


async def run_agent(question: str, mcp: MCPClient, max_steps: int = 8):
    console.print(Panel(f"[bold cyan]Question:[/] {question}", expand=False))

    tool_descriptions = mcp.get_tool_descriptions()
    system = build_system_prompt(tool_descriptions)
    messages = [{"role": "user", "content": question}]
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
                nudge = (
                    "You must use at least one tool before Final Answer. "
                    "Use Action + Action Input (JSON) format."
                )
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": nudge})
                continue
            console.print(Panel(
                f"[bold green]Final Answer:[/]\n{parsed[1]}",
                title="✓ Complete", expand=False
            ))
            return parsed[1]

        # ── Tool call (parsed JSON) ──
        elif parsed[0] == "action":
            _, tool_name, tool_args = parsed
            console.print(f"[yellow]→ MCP call: {tool_name}({tool_args})[/yellow]")
            obs = await mcp.call_tool(tool_name, tool_args)
            tools_used.append(tool_name)
            console.print(f"[green]Observation: {obs[:300]}[/green]")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        # ── Tool call (raw string fallback — infer schema) ──
        elif parsed[0] == "action_raw":
            _, tool_name, raw_input = parsed
            tool_schema = next(
                (t for t in mcp._all_tools if t["name"] == tool_name), None
            )
            if tool_schema:
                props = tool_schema.get("input_schema", {}).get("properties", {})
                first_key = next(iter(props), "input")
                tool_args = {first_key: raw_input}
            else:
                tool_args = {"input": raw_input}

            console.print(f"[yellow]→ MCP call (inferred): {tool_name}({tool_args})[/yellow]")
            obs = await mcp.call_tool(tool_name, tool_args)
            tools_used.append(tool_name)
            console.print(f"[green]Observation: {obs[:300]}[/green]")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        # ── Parse failure ──
        else:
            tool_names = [t["name"] for t in mcp._all_tools]
            nudge = (
                f"Format error. Use exactly:\n"
                f"Thought: ...\n"
                f"Action: <one of: {', '.join(tool_names)}>\n"
                f'Action Input: {{"key": "value"}}\n\n'
                f"Never write Action and Final Answer in the same response."
            )
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": nudge})

    return "Agent reached max steps without a Final Answer."


async def main():
    console.print("[bold]Initializing MCP client...[/bold]")
    mcp = MCPClient()
    await mcp.initialize()

    questions = [
        "What was the Phase 2 RAGAS capstone score?",
        "What is the percentage gain from Phase 1 (0.638) to Phase 2 (0.827)?",
        "Fetch the content from https://www.anthropic.com/research and summarise what you find.",
    ]

    for q in questions:
        await run_agent(q, mcp)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())