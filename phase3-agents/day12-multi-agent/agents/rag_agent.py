# phase3-agents/day12-multi-agent/agents/rag_agent.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "day10-react-from-scratch"))
from tools import rag_search, dictionary
from .base_agent import BaseAgent

class RAGAgent(BaseAgent):
    name = "rag"
    tools = {"rag_search": rag_search, "dictionary": dictionary}

    def system_prompt(self) -> str:
        return f"""You are a specialist RAG agent. You answer questions using the knowledge base.

        Tools: {", ".join(self.tools.keys())}
        {self._base_rules()}
        Never invent facts. Only use what the tools return."""