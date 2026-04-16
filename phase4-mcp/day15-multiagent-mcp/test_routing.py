# phase4-mcp/day15-multiagent-mcp/test_routing.py
from supervisor import route

cases = [
    ("What is semantic chunking?",               "rag"),
    ("What is 144 divided by 12?",               "math"),
    ("Store the fact that RAGAS uses 4 metrics", "memory"),
    ("What did I store about RAGAS?",            "memory"),
    ("Explain cross-encoder reranking",          "rag"),
    ("Calculate 2 to the power of 10",           "math"),
]

passed = 0
for question, expected in cases:
    got = route(question)
    status = "✓" if got == expected else "✗"
    if got == expected:
        passed += 1
    print(f"{status} '{question[:45]}' → {got} (expected {expected})")

print(f"\n{passed}/{len(cases)} routing tests passed")