# phase5-production/day17-fastapi/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "phase4-mcp" / "day15-multiagent-mcp"))
sys.path.insert(0, str(REPO_ROOT / "phase3-agents" / "day11-memory"))

from client.mcp_client import MCPClient
from memory import LongTermMemory

from models import AskRequest, AskResponse, HealthResponse
from pipeline import run_pipeline
from memory import LongTermMemory, init_db

# Shared state loaded once at startup
app_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MCP probe and memory at startup."""
    
    probe = MCPClient()
    await probe.initialize()
    app_state["tools"] = probe._local_tools + probe._fetch_tools   # returns List[str]
    app_state["servers"] = ["local", "fetch"]
    
    init_db()
    memory = LongTermMemory()
    app_state["memory"] = memory
    yield
    # shutdown — nothing to teardown

app = FastAPI(title="AI Engineering Journey API", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health():
    memory: LongTermMemory = app_state["memory"]
    entries = memory.count()   # we'll add this method below
    return HealthResponse(
        status="ok",
        mcp_servers=app_state["servers"],
        tools_available=len(app_state["tools"]),
        memory_entries=entries,
    )

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    try:
        result = await run_pipeline(request.question)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))