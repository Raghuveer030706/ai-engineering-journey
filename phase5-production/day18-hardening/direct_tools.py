# phase5-production/day18-hardening/direct_tools.py
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase3-agents" / "day11-memory"))

from memory import Memory

import chromadb
from sentence_transformers import SentenceTransformer

# ── Shared instances (loaded once at import, not per call) ────────────────────
_memory = Memory()

CHROMA_PATH = REPO_ROOT / "phase2-advanced-rag" / "day9-capstone" / "chroma_db"
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
_collection = _chroma_client.get_collection("day9")

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

# ── Tool functions ────────────────────────────────────────────────────────────

def calculator(expression: str) -> str:
    try:
        allowed = {k: v for k, v in math.__dict__.items()
                   if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def project_facts(key: str) -> str:
    key = key.strip()
    result = PROJECT_FACTS.get(key)
    if result:
        return result
    return f"Key '{key}' not found. Available: {', '.join(PROJECT_FACTS.keys())}"

def rag_search(query: str) -> str:
    try:
        embedding = _embed_model.encode([query]).tolist()
        results = _collection.query(query_embeddings=embedding, n_results=3)
        docs = results["documents"][0] if results["documents"] else []
        if not docs:
            return "No results found."
        return "\n\n---\n\n".join(docs)
    except Exception as e:
        return f"RAG error: {e}"

def memory_store(entry: str) -> str:
    parts = [p.strip() for p in entry.split("|")]
    if len(parts) < 2:
        return "ERROR: format must be key|value or key|value|confidence"
    key, value = parts[0], parts[1]
    confidence = float(parts[2]) if len(parts) >= 3 else 0.9
    _memory.store(key, value, confidence=confidence, source="api_agent")
    return f"Stored: '{key}' = '{value}' [conf={confidence}]"

def memory_retrieve(key: str) -> str:
    result = _memory.retrieve(key.strip())
    if result is None:
        return f"No memory found for key: '{key}'"
    return f"[{result.get('tier','?')}-term, conf={result.get('confidence','?')}] {result['value']}"

# ── Unified call interface (mirrors MCPClient.call_tool signature) ────────────
TOOL_MAP = {
    "calculator":     lambda args: calculator(args["expression"]),
    "project_facts":  lambda args: project_facts(args["key"]),
    "rag_search":     lambda args: rag_search(args["query"]),
    "memory_store":   lambda args: memory_store(args["entry"]),
    "memory_retrieve": lambda args: memory_retrieve(args["key"]),
}

def call_tool_direct(tool_name: str, arguments: dict) -> str:
    fn = TOOL_MAP.get(tool_name)
    if fn:
        return fn(arguments)
    return f"Unknown tool: {tool_name}"