# phase3-agents/day13-capstone/planner.py
import json
import anthropic
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

PLANNER_PROMPT = """You are a planner that decomposes user questions into ordered sub-tasks.

Each sub-task must be handled by exactly one of these agents:
- rag    : retrieve facts, concepts, scores, or data from the knowledge base
- math   : arithmetic, calculations, numeric problems
- memory : recalling a fact the agent explicitly stored in a previous session,
           or storing a new fact for future sessions

CRITICAL RULES for choosing the right agent:
- Use rag when the question asks WHAT something is, HOW something works,
  or asks for scores, results, or data from documents or research
- Use memory ONLY when the question says "recall", "retrieve from memory",
  "what did I store", or explicitly references a previously saved key
- Never use memory to answer factual questions — those always go to rag
- If one sub-task can answer multiple related facts from the same source,
  keep it as ONE sub-task, do not split unnecessarily
- Only create multiple sub-tasks when parts genuinely need different agents

Order sub-tasks so later ones can use results from earlier ones.
Each sub-task must be self-contained and answerable independently.

Return ONLY valid JSON. No explanation. No markdown. No backticks.

Format:
{
  "sub_tasks": [
    {"id": 1, "agent": "rag",    "task": "what is semantic chunking and how many chunks did Day 3 produce"},
    {"id": 2, "agent": "math",   "task": "calculate 104 multiplied by 0.827"},
    {"id": 3, "agent": "memory", "task": "store: phase3_score|0.827|0.95"}
  ]
}"""

# phase3-agents/day13-capstone/planner.py

def plan(question: str) -> list[dict]:
    for attempt in range(2):  # try twice before fallback
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=PLANNER_PROMPT,
            messages=[{"role": "user", "content": question}],
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            data = json.loads(raw)
            sub_tasks = data.get("sub_tasks", [])
            if sub_tasks:
                return sub_tasks
        except json.JSONDecodeError:
            if attempt == 0:
                print(f"Planner parse failed on attempt 1, retrying...")
                continue

    # Both attempts failed — fail loudly, do not silently route wrong
    print(f"Planner failed after 2 attempts. Defaulting to rag.")
    print(f"Question was: {question}")
    return [{"id": 1, "agent": "rag", "task": question}]