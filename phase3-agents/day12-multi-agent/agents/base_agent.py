# phase3-agents/day12-multi-agent/agents/base_agent.py
import re
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
    Shared ReAct loop. All specialist agents inherit this.
    Subclasses define: name, tools, system_prompt.
    """

    name: str = "base"
    tools: dict = {}
    max_steps: int = 6

    # Add this method inside the BaseAgent class
    def _base_rules(self) -> str:
        tool_names = ", ".join(self.tools.keys())
        return f"""
        CRITICAL RULES:
        - Write ONLY ONE block per response: either a tool call OR a Final Answer. Never both.
        - You will receive an Observation after each tool call. Wait for it.
        - Only write Final Answer AFTER you have received at least one real Observation.
        - Never invent an Observation. Never skip waiting for one.

        Available tools: {tool_names}

        Correct format for a tool call:
        Thought: <your reasoning>
        Action: <tool name>
        Action Input: <input>

        Correct format for final answer (only after an Observation):
        Final Answer: <your answer>
        """

    def system_prompt(self) -> str:
        raise NotImplementedError

    def run(self, task: str) -> str:
        tool_names = ", ".join(self.tools.keys())
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
                    # Model skipped tool call — force it to call the tool first
                    nudge = (
                        f"You wrote a Final Answer without calling any tool first. "
                        f"You MUST call a tool and wait for the Observation before answering. "
                        f"Start again with ONLY:\n"
                        f"Thought: <your reasoning>\n"
                        f"Action: <one of {tool_names}>\n"
                        f"Action Input: <input>\n"
                        f"Do NOT write Final Answer yet."
                    )
                    messages.append({"role": "assistant", "content": text})
                    messages.append({"role": "user", "content": nudge})
                    continue

                console.print(f"[dim]{self.name} complete. Tools used: {tools_used}[/dim]")
                return parsed[1]

            # ── Tool call ──
            elif parsed[0] == "action":
                _, tool_name, tool_input = parsed
                tool_name_clean = tool_name.lower().strip()

                if tool_name_clean not in self.tools:
                    obs = (
                        f"ERROR: '{tool_name}' not available to {self.name} agent. "
                        f"Your tools: {tool_names}"
                    )
                else:
                    console.print(
                        f"[yellow]{self.name} → {tool_name_clean}({tool_input!r})[/yellow]"
                    )
                    try:
                        obs = self.tools[tool_name_clean](tool_input)
                    except Exception as e:
                        obs = f"Tool error: {e}"
                    tools_used.append(tool_name_clean)

                console.print(f"[green]Observation: {obs}[/green]")
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": f"Observation: {obs}"})

            # ── Parse failure — model wrote Action+FinalAnswer together ──
            else:
                nudge = (
                    f"Format error. You must write ONLY ONE of these per response:\n\n"
                    f"Option 1 — call a tool:\n"
                    f"Thought: <reasoning>\n"
                    f"Action: <one of {tool_names}>\n"
                    f"Action Input: <input>\n\n"
                    f"Option 2 — after you have received an Observation:\n"
                    f"Final Answer: <answer>\n\n"
                    f"Never write Action and Final Answer in the same response."
                )
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": nudge})

        return f"{self.name} agent reached max steps without Final Answer."

    def _parse(self, text: str):
        if "Final Answer:" in text:
            return ("final", text.split("Final Answer:")[-1].strip())
        action_match = re.search(r"Action:\s*(.+)", text)
        input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
        if action_match and input_match:
            return ("action", action_match.group(1).strip(), input_match.group(1).strip())
        return ("unknown", text)