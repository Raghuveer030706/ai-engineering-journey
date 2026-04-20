# phase5-production/day17-fastapi/models.py
from pydantic import BaseModel
from typing import List, Optional

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sub_tasks: int
    agents_used: List[str]
    llm_calls: int
    duration_ms: int
    estimated_cost_usd: float

class MCPServerStatus(BaseModel):
    name: str
    status: str
    tools: List[str]

class HealthResponse(BaseModel):
    status: str
    mcp_servers: List[str]
    tools_available: int
    memory_entries: int