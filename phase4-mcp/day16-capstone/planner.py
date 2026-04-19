# phase4-mcp/day16-capstone/planner.py
import json
import anthropic
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

PLANNER_PROMPT = """You are a planner that decomposes user questions into ordered sub-tasks.

Each sub-task must be handled by exactly one of these agents:
- rag    : facts, concepts, scores from the knowledge base or project documents
- math   : arithmetic, calculations, numeric problems
- memory : storing a fact for later OR recalling a previously stored fact
- fetch  : reading live content from a public URL

CRITICAL RULES for choosing the correct agent:
- Use rag when the question asks WHAT something is, HOW something works,
  or asks for scores and results from the project knowledge base
- Use math when the question requires a calculation with numbers
- Use memory ONLY when the question says "recall", "retrieve from memory",
  "what did I store", or explicitly references a previously saved key
- Use fetch ONLY when the question references a specific URL to read
- Never use memory to answer factual questions — those always go to rag
- If one sub-task can answer multiple related facts from the same source,
  keep it as ONE sub-task — do not split unnecessarily
- Order sub-tasks so later ones can use results from earlier ones

Return ONLY valid JSON. No explanation. No markdown. No backticks.

Format:
{
  "sub_tasks": [
    {"id": 1, "agent": "rag",    "task": "what was the Phase 2 RAGAS score"},
    {"id": 2, "agent": "math",   "task": "multiply the Phase 2 score by 100"},
    {"id": 3, "agent": "memory", "task": "store: phase2_pct|82.7|0.95"},
    {"id": 4, "agent": "fetch",  "task": "fetch https://example.com and summarise"}
  ]
}"""


def plan(question: str) -> list[dict]:
    for attempt in range(2):
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

    # Both attempts failed — fallback loudly
    print(f"Planner failed after 2 attempts. Defaulting to rag.")
    print(f"Question was: {question}")
    return [{"id": 1, "agent": "rag", "task": question}]