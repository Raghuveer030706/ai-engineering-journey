# phase5-production/day17-fastapi/pipeline.py
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

# Direct path inserts — bypasses hyphenated folder name problem
sys.path.insert(0, str(REPO_ROOT / "phase4-mcp" / "day16-capstone"))
sys.path.insert(0, str(REPO_ROOT / "phase4-mcp" / "day15-multiagent-mcp"))

from planner import plan
from orchestrator import orchestrate
from synthesizer import synthesize
from client.mcp_client import MCPClient

COST_PER_TOKEN = (0.80 + 4.00) / 2 / 1_000_000  # avg haiku input+output
AVG_TOKENS_PER_CALL = 600

async def run_pipeline(question: str) -> dict:
    start = time.time()

    mcp_client = MCPClient()
    await mcp_client.initialize()

    # orchestrate() returns List[dict], each dict has keys:
    # id, agent, task, result
    results = await orchestrate(question, mcp=mcp_client)

    # Extract metadata from results — no orchestrator changes needed
    agents_used = list({r["agent"] for r in results})
    sub_task_count = len(results)

    # Conservative llm_calls estimate:
    # 1 planner + (sub_tasks * 3 avg steps) + 1 synthesizer
    llm_calls = 1 + (sub_task_count * 3) + 1

    answer = synthesize(question, results)

    duration_ms = int((time.time() - start) * 1000)
    estimated_cost = llm_calls * AVG_TOKENS_PER_CALL * COST_PER_TOKEN

    return {
        "answer": answer,
        "sub_tasks": sub_task_count,
        "agents_used": agents_used,
        "llm_calls": llm_calls,
        "duration_ms": duration_ms,
        "estimated_cost_usd": round(estimated_cost, 6),
    }