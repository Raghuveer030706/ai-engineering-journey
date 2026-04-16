# phase4-mcp/day15-multiagent-mcp/agents/math_agent.py
from .base_agent import BaseAgent

class MathAgent(BaseAgent):
    name = "math"
    allowed_tools = ["calculator"]

    def system_prompt(self) -> str:
        return f"""You are a specialist math agent. You solve calculations precisely.
{self._base_rules()}
Show your working. Never guess a number."""