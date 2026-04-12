# phase3-agents/day12-multi-agent/run.py
from supervisor import run

# run.py — change the recall question to be key-specific
questions = [
    "What is the difference between HyDE and naive RAG retrieval?",
    "What is 1024 divided by 32, then multiplied by 7?",
    "Store this fact in memory with key 'phase2_score': the best Phase 2 score was 0.827",
    "Use the memory_retrieve tool with key 'phase2_score' and tell me what it says.",  # ← explicit key
]

for q in questions:
    run(q)
    print("\n" + "="*60 + "\n")