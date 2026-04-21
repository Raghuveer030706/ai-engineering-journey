# phase5-production/day18-hardening/main.py
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase4-mcp" / "day15-multiagent-mcp"))
sys.path.insert(0, str(REPO_ROOT / "phase3-agents" / "day11-memory"))

from client.mcp_client import MCPClient
from memory import LongTermMemory, init_db
from models import AskRequest, AskResponse, HealthResponse
from pipeline import run_pipeline, HybridClient

app_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # MCP init once — not per request
    mcp = MCPClient()
    await mcp.initialize()
    hybrid = HybridClient(mcp_fallback=mcp)
    app_state["hybrid"] = hybrid
    app_state["tools"] = hybrid._all_tools
    app_state["servers"] = ["local", "fetch"]

    init_db()
    app_state["memory"] = LongTermMemory()
    yield

app = FastAPI(title="AI Engineering Journey API", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        mcp_servers=app_state["servers"],
        tools_available=len(app_state["tools"]),
        memory_entries=0,
    )

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    try:
        result = await run_pipeline(request.question, app_state["hybrid"])
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))