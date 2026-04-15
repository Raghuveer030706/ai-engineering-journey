# phase4-mcp/day14-mcp-foundations/server/local_server.py
import math
import json
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("local-tools")

PROJECT_FACTS = {
    "phase1_score":       "Phase 1 naive RAG keyword hit rate: 6/6 = 100%",
    "phase2_baseline":    "Phase 2 RAGAS baseline Day 6: 0.638",
    "phase2_hybrid":      "Phase 2 Day 7 hybrid retrieval: 0.807",
    "phase2_capstone":    "Phase 2 Day 9 capstone: 0.827",
    "phase2_gain":        "Phase 2 improvement: +0.189 absolute, +29.6% percentage",
    "day3_chunks":        "Day 3 chunk counts: 180 raw → 104 clean. 42.2% removed.",
    "day10_agent":        "Day 10 ReAct agent: 3 tools, 4 steps, 5 LLM calls",
    "day11_memory":       "Day 11 memory: short-term RAM + long-term SQLite, confidence scores",
    "day12_multiagent":   "Day 12 multi-agent: supervisor + 3 specialists, ~5 LLM calls",
    "day13_capstone":     "Day 13 capstone: planner + multi-path routing, ~11 LLM calls",
}


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="calculator",
            description="Evaluate a safe mathematical expression. Input must be a valid Python math expression using only numbers and operators.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate e.g. '(0.827 - 0.638) / 0.638 * 100'",
                    }
                },
                "required": ["expression"],
            },
        ),
        types.Tool(
            name="project_facts",
            description="Look up a fact about the AI engineering journey project by key. Returns the stored fact string.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Fact key to look up. Available keys: " + ", ".join(PROJECT_FACTS.keys()),
                    }
                },
                "required": ["key"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "calculator":
        expression = arguments.get("expression", "")
        try:
            allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    elif name == "project_facts":
        key = arguments.get("key", "").strip()
        result = PROJECT_FACTS.get(key)
        if result:
            return [types.TextContent(type="text", text=result)]
        available = ", ".join(PROJECT_FACTS.keys())
        return [types.TextContent(
            type="text",
            text=f"Key '{key}' not found. Available keys: {available}"
        )]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())