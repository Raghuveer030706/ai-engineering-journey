# phase5-production/day18-hardening/logger.py
import json
import time
from pathlib import Path
from datetime import datetime

LOG_PATH = Path(__file__).parent / "requests.log"

class RequestLogger:
    def __init__(self, question: str):
        self.question = question
        self.request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.stages: list[dict] = []
        self.start = time.time()

    def log_stage(self, stage: str, detail: str, duration_ms: int):
        self.stages.append({
            "stage": stage,
            "detail": detail,
            "duration_ms": duration_ms,
        })

    def write(self, response: dict):
        record = {
            "request_id": self.request_id,
            "timestamp": datetime.now().isoformat(),
            "question": self.question,
            "total_ms": int((time.time() - self.start) * 1000),
            "stages": self.stages,
            "response_summary": {
                "agents_used": response.get("agents_used"),
                "llm_calls": response.get("llm_calls"),
                "estimated_cost_usd": response.get("estimated_cost_usd"),
            },
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        return self.request_id