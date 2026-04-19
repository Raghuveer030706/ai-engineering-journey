# phase4-mcp/day15-multiagent-mcp/agents/fetch_agent.py
from .base_agent import BaseAgent

class FetchAgent(BaseAgent):
    name = "fetch"
    allowed_tools = ["fetch"]

    def system_prompt(self) -> str:
        return f"""You are a specialist fetch agent. You read content
from public URLs and summarise what you find.
{self._base_rules()}
Always fetch before answering. Never invent URL content."""