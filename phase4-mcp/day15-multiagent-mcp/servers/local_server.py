# phase4-mcp/day15-multiagent-mcp/servers/local_server.py
import math
import sys
import asyncio
from pathlib import Path

# Pull in Day 11 memory and Day 9 ChromaDB
sys.path.append(str(Path(__file__).parent.parent.parent.parent /
                    "phase3-agents" / "day11-memory"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent /
                    "phase3-agents" / "day10-react-from-scratch"))

from memory import Memory
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

import chromadb
from sentence_transformers import SentenceTransformer

app = Server("local-tools")

# ── Shared instances ──────────────────────────────────────────────────────────
_memory = Memory()

CHROMA_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "phase2-advanced-rag" / "day9-capstone" / "chroma_db"
)
COLLECTION_NAME = "day9"
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
_collection = _chroma_client.get_collection(COLLECTION_NAME)

PROJECT_FACTS = {
    "phase1_score":     "Phase 1 naive RAG keyword hit rate: 6/6 = 100%",
    "phase2_baseline":  "Phase 2 RAGAS baseline Day 6: 0.638",
    "phase2_hybrid":    "Phase 2 Day 7 hybrid retrieval: 0.807",
    "phase2_capstone":  "Phase 2 Day 9 capstone: 0.827",
    "phase2_gain":      "Phase 2 improvement: +0.189 absolute, +29.6% percentage",
    "day3_chunks":      "Day 3 chunk counts: 180 raw → 104 clean. 42.2% removed.",
    "day10_agent":      "Day 10 ReAct agent: 3 tools, 4 steps, 5 LLM calls",
    "day11_memory":     "Day 11 memory: short-term RAM + long-term SQLite, confidence scores",
    "day12_multiagent": "Day 12 multi-agent: supervisor + 3 specialists, ~5 LLM calls",
    "day13_capstone":   "Day 13 capstone: planner + multi-path routing, ~11 LLM calls",
}


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="calculator",
            description="Evaluate a safe mathematical expression.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Valid Python math expression e.g. '(0.827 - 0.638) / 0.638 * 100'",
                    }
                },
                "required": ["expression"],
            },
        ),
        types.Tool(
            name="project_facts",
            description="Look up a project fact by key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Fact key. Available: " + ", ".join(PROJECT_FACTS.keys()),
                    }
                },
                "required": ["key"],
            },
        ),
        types.Tool(
            name="rag_search",
            description="Search the AI engineering project knowledge base for facts, concepts, and scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    }
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="memory_store",
            description="Store a fact in memory for future sessions. Format: key|value|confidence",
            inputSchema={
                "type": "object",
                "properties": {
                    "entry": {
                        "type": "string",
                        "description": "Format: key|value|confidence e.g. 'phase2_score|0.827|0.95'",
                    }
                },
                "required": ["entry"],
            },
        ),
        types.Tool(
            name="memory_retrieve",
            description="Retrieve a previously stored fact from memory by key.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key to retrieve",
                    }
                },
                "required": ["key"],
            },
        ),
    ]


# ── Tool implementations ──────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    if name == "calculator":
        expression = arguments.get("expression", "")
        try:
            allowed = {k: v for k, v in math.__dict__.items()
                       if not k.startswith("__")}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]

    elif name == "project_facts":
        key = arguments.get("key", "").strip()
        result = PROJECT_FACTS.get(key)
        if result:
            return [types.TextContent(type="text", text=result)]
        return [types.TextContent(
            type="text",
            text=f"Key '{key}' not found. Available: {', '.join(PROJECT_FACTS.keys())}"
        )]

    elif name == "rag_search":
        query = arguments.get("query", "").strip()
        try:
            embedding = _embed_model.encode([query]).tolist()
            results = _collection.query(
                query_embeddings=embedding,
                n_results=3,
            )
            docs = results["documents"][0] if results["documents"] else []
            if not docs:
                return [types.TextContent(type="text", text="No results found.")]
            combined = "\n\n---\n\n".join(docs)
            return [types.TextContent(type="text", text=combined)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"RAG error: {e}")]

    elif name == "memory_store":
        entry = arguments.get("entry", "")
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) < 2:
            return [types.TextContent(
                type="text",
                text="ERROR: format must be key|value or key|value|confidence"
            )]
        key = parts[0]
        value = parts[1]
        confidence = float(parts[2]) if len(parts) >= 3 else 0.9
        _memory.store(key, value, confidence=confidence, source="mcp_agent")
        return [types.TextContent(
            type="text",
            text=f"Stored: '{key}' = '{value}' [conf={confidence}]"
        )]

    elif name == "memory_retrieve":
        key = arguments.get("key", "").strip()
        result = _memory.retrieve(key)
        if result is None:
            return [types.TextContent(
                type="text", text=f"No memory found for key: '{key}'"
            )]
        tier = result.get("tier", "?")
        conf = result.get("confidence", "?")
        return [types.TextContent(
            type="text",
            text=f"[{tier}-term, conf={conf}] {result['value']}"
        )]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream, write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())