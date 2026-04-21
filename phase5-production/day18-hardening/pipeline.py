# phase5-production/day18-hardening/pipeline.py
import sys
import time
import asyncio
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase4-mcp" / "day16-capstone"))
sys.path.insert(0, str(REPO_ROOT / "phase4-mcp" / "day15-multiagent-mcp"))

from planner import plan
from orchestrator import orchestrate
from synthesizer import synthesize
from client.mcp_client import MCPClient
from logger import RequestLogger
from cost import CostTracker
from direct_tools import call_tool_direct, TOOL_MAP

TASK_TIMEOUT_SECONDS = 45

class HybridClient:
    """
    Drop-in replacement for MCPClient.
    Local tools → direct Python call (no subprocess).
    Fetch tool  → MCP subprocess (genuinely external).
    Same call_tool() signature so agents need zero changes.
    """
    def __init__(self, mcp_fallback: MCPClient):
        self._mcp = mcp_fallback
        self._local_tool_names = list(TOOL_MAP.keys())
        # Expose same attributes agents read
        self._all_tools = self._build_tool_schemas()

    def _build_tool_schemas(self) -> list[dict]:
        schemas = [
            {"name": n, "description": "", "input_schema": {}, "server": "direct"}
            for n in self._local_tool_names
        ]
        # Append fetch tools from real MCP client
        fetch = [t for t in self._mcp._all_tools if t["server"] == "fetch"]
        return schemas + fetch

    def get_tool_descriptions(self) -> str:
        lines = [f"- {t['name']} [{t['server']}]" for t in self._all_tools]
        return "\n".join(lines)

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name in self._local_tool_names:
            # Direct call — no subprocess, no MCP overhead
            return call_tool_direct(tool_name, arguments)
        else:
            # Fetch or unknown — fall back to MCP
            return await self._mcp.call_tool(tool_name, arguments)


async def run_pipeline(question: str, hybrid: HybridClient) -> dict:
    logger = RequestLogger(question)
    tracker = CostTracker()

    # No MCP init here — hybrid already warm from startup
    t0 = time.time()
    sub_tasks = plan(question)
    tracker.llm_calls += 1
    logger.log_stage("planner", f"{len(sub_tasks)} sub-tasks", int((time.time() - t0) * 1000))

    t0 = time.time()
    results = await asyncio.wait_for(
        orchestrate(question, mcp=hybrid),
        timeout=TASK_TIMEOUT_SECONDS * max(len(sub_tasks), 1),
    )
    agents_used = list({r["agent"] for r in results})
    tracker.llm_calls += len(sub_tasks) * 3
    logger.log_stage("orchestrator", f"agents: {agents_used}", int((time.time() - t0) * 1000))

    t0 = time.time()
    answer = synthesize(question, results)
    tracker.llm_calls += 1
    logger.log_stage("synthesizer", "done", int((time.time() - t0) * 1000))

    cost = tracker.summary()
    response = {
        "answer": answer,
        "sub_tasks": len(sub_tasks),
        "agents_used": agents_used,
        "llm_calls": cost["llm_calls"],
        "duration_ms": int((time.time() - logger.start) * 1000),
        "estimated_cost_usd": cost["estimated_cost_usd"],
        "request_id": "",
    }
    response["request_id"] = logger.write(response)
    return response