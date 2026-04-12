# phase3-agents/day12-multi-agent/agents/memory_agent.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "day11-memory"))
from memory import Memory
from .base_agent import BaseAgent

_memory = Memory()

def memory_store(key_value: str) -> str:
    parts = [p.strip() for p in key_value.split("|")]
    if len(parts) < 2:
        return "ERROR: format must be key|value or key|value|confidence"
    key, value = parts[0], parts[1]
    confidence = float(parts[2]) if len(parts) >= 3 else 0.9
    _memory.store(key, value, confidence=confidence, source="memory_agent")
    return f"Stored: '{key}' = '{value}' [conf={confidence}]"

def memory_retrieve(key: str) -> str:
    result = _memory.retrieve(key.strip())
    if result is None:
        return f"No memory found for: '{key}'"
    return f"[{result.get('tier','?')}-term, conf={result.get('confidence','?')}] {result['value']}"

class MemoryAgent(BaseAgent):
    name = "memory"
    max_steps = 10          # ← add this line
    tools = {"memory_store": memory_store, "memory_retrieve": memory_retrieve}

    def system_prompt(self) -> str:
        mem_context = _memory.context_block()
        return f"""You are a specialist memory agent. You store and retrieve facts.

        Current memory state:
        {mem_context}

        Tools: {", ".join(self.tools.keys())}
        memory_store input format: key|value|confidence (confidence optional, default 0.9)
        {self._base_rules()}"""