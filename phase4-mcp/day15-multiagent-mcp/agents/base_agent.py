# phase4-mcp/day15-multiagent-mcp/agents/base_agent.py
import re
import json
import anthropic
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / ".env")

console = Console()
client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"


class BaseAgent:
    """
    ReAct loop wired to MCPClient.
    Subclasses define: name, allowed_tools, system_prompt().
    """
    name: str = "base"
    allowed_tools: list[str] = []
    max_steps: int = 6

    def __init__(self, mcp):
        self.mcp = mcp

    def system_prompt(self) -> str:
        raise NotImplementedError

    def _base_rules(self) -> str:
        tools = ", ".join(self.allowed_tools)
        return f"""
CRITICAL RULES:
- Write ONLY ONE block per response: either a tool call OR a Final Answer. Never both.
- Wait for the Observation after each tool call before continuing.
- Only write Final Answer AFTER receiving at least one real Observation.
- Never invent an Observation.

Available tools: {tools}

Tool call format:
Thought: <reasoning>
Action: <tool name>
Action Input: {{"key": "value"}}

Final answer format (only after Observation):
Final Answer: <answer>
"""

    def _parse(self, text: str):
        if "Final Answer:" in text:
            lines = text.split("\n")
            has_action = any(l.strip().startswith("Action:") for l in lines)
            if not has_action:
                return ("final", text.split("Final Answer:")[-1].strip())

        action_match = re.search(r"Action:\s*(.+)", text)
        if not action_match:
            return ("unknown", text)

        tool_name = action_match.group(1).strip()

        input_match = re.search(r"Action Input:\s*(\{[\s\S]*?\})(?:\n|$)", text)
        if input_match:
            try:
                tool_args = json.loads(input_match.group(1).strip())
                return ("action", tool_name, tool_args)
            except json.JSONDecodeError:
                pass

        raw_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
        if raw_match:
            raw = raw_match.group(1).strip()
            try:
                tool_args = json.loads(raw)
                return ("action", tool_name, tool_args)
            except json.JSONDecodeError:
                return ("action_raw", tool_name, raw)

        return ("unknown", text)

    async def run(self, task: str) -> str:
        messages = [{"role": "user", "content": task}]
        tools_used = []
        step = 0

        console.print(f"\n[dim]── {self.name} agent started ──[/dim]")

        while step < self.max_steps:
            step += 1

            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=self.system_prompt(),
                messages=messages,
            )
            text = response.content[0].text
            console.print(f"[dim]{self.name}>[/dim] {text}")

            parsed = self._parse(text)

            # ── Final Answer ──
            if parsed[0] == "final":
                if not tools_used:
                    nudge = (
                        f"You must use at least one tool before Final Answer. "
                        f"Do NOT write Final Answer yet.\n"
                        f"Available tools: {', '.join(self.allowed_tools)}"
                    )
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": nudge})
                    continue
                console.print(f"[dim]{self.name} complete. Tools: {tools_used}[/dim]")
                return parsed[1]

            # ── Tool call (JSON) ──
            elif parsed[0] == "action":
                _, tool_name, tool_args = parsed
                if tool_name not in self.allowed_tools:
                    obs = (
                        f"ERROR: '{tool_name}' not available to {self.name} agent. "
                        f"Your tools: {', '.join(self.allowed_tools)}"
                    )
                else:
                    console.print(
                        f"[yellow]{self.name} → MCP: {tool_name}({tool_args})[/yellow]"
                    )
                    obs = await self.mcp.call_tool(tool_name, tool_args)
                    tools_used.append(tool_name)
                console.print(f"[green]Observation: {str(obs)[:300]}[/green]")
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Observation: {obs}"})

            # ── Tool call (raw fallback — infer schema) ──
            elif parsed[0] == "action_raw":
                _, tool_name, raw_input = parsed
                tool_schema = next(
                    (t for t in self.mcp._all_tools if t["name"] == tool_name), None
                )
                if tool_schema:
                    props = tool_schema.get("input_schema", {}).get("properties", {})
                    first_key = next(iter(props), "input")
                    tool_args = {first_key: raw_input}
                else:
                    tool_args = {"input": raw_input}

                console.print(
                    f"[yellow]{self.name} → MCP (inferred): {tool_name}({tool_args})[/yellow]"
                )
                obs = await self.mcp.call_tool(tool_name, tool_args)
                tools_used.append(tool_name)
                console.print(f"[green]Observation: {str(obs)[:300]}[/green]")
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Observation: {obs}"})

            # ── Parse failure ──
            else:
                nudge = (
                    f"Format error. Write ONLY ONE of:\n\n"
                    f"Option 1 — tool call:\n"
                    f"Thought: ...\n"
                    f"Action: <one of: {', '.join(self.allowed_tools)}>\n"
                    f'Action Input: {{"key": "value"}}\n\n'
                    f"Option 2 — after Observation:\n"
                    f"Final Answer: <answer>\n\n"
                    f"Never write Action and Final Answer in the same response."
                )
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": nudge})

        return f"{self.name} agent reached max steps without Final Answer."