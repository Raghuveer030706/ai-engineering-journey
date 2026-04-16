# phase4-mcp/day15-multiagent-mcp/agents/memory_agent.py
from .base_agent import BaseAgent

class MemoryAgent(BaseAgent):
    name = "memory"
    allowed_tools = ["memory_store", "memory_retrieve"]

    def system_prompt(self) -> str:
        return f"""You are a specialist memory agent. You store and retrieve facts.
memory_store entry format: key|value|confidence (confidence optional, default 0.9)
{self._base_rules()}"""