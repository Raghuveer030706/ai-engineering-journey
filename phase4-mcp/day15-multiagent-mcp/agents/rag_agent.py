# phase4-mcp/day15-multiagent-mcp/agents/rag_agent.py
from .base_agent import BaseAgent

class RAGAgent(BaseAgent):
    name = "rag"
    allowed_tools = ["rag_search", "project_facts"]

    def system_prompt(self) -> str:
        return f"""You are a specialist RAG agent. You answer questions
using the knowledge base and project facts.
{self._base_rules()}
Never invent facts. Only use what tools return."""