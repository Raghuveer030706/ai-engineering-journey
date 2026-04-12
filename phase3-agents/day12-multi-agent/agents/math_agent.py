# phase3-agents/day12-multi-agent/agents/math_agent.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "day10-react-from-scratch"))
from tools import calculator
from .base_agent import BaseAgent

class MathAgent(BaseAgent):
    name = "math"
    tools = {"calculator": calculator}

    def system_prompt(self) -> str:
        return f"""You are a specialist math agent. You solve calculations precisely.

        Tools: {", ".join(self.tools.keys())}
        {self._base_rules()}
        Show your working. Never guess a number."""