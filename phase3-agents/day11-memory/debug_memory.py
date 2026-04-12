# phase3-agents/day11-memory/debug_memory.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from memory import Memory

m = Memory()

# Manually store the Phase 2 score
m.store("phase2_score", "0.827", confidence=0.95, source="manual")
print("Stored.")

# Immediately retrieve
result = m.retrieve("phase2_score")
print(f"Retrieved: {result}")

# Check long-term directly
lt = m.long.retrieve("phase2_score")
print(f"Long-term: {lt}")

# Print full context block
print(m.context_block())