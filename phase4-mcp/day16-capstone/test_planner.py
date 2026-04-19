# phase4-mcp/day16-capstone/test_planner.py
from planner import plan

cases = [
    {
        "question": "What is HyDE?",
        "expected_agents": ["rag"],
        "expected_count": 1,
    },
    {
        "question": "What was the Phase 2 score and what is that as a percentage gain from 0.638?",
        "expected_agents": ["rag", "math"],
        "expected_count": 2,
    },
    {
        "question": "What was the Day 9 RAGAS score, multiply by 100, store as 'day9_pct'",
        "expected_agents": ["rag", "math", "memory"],
        "expected_count": 3,
    },
    {
        "question": "Fetch https://www.anthropic.com and summarise it",
        "expected_agents": ["fetch"],
        "expected_count": 1,
    },
    {
        "question": "What was the Phase 2 score, calculate percentage gain, "
                    "store as 'phase2_gain', then fetch https://anthropic.com",
        "expected_agents": ["rag", "math", "memory", "fetch"],
        "expected_count": 4,
    },
]

passed = 0
for case in cases:
    sub_tasks = plan(case["question"])
    agents_returned = [st["agent"] for st in sub_tasks]
    count_ok = len(sub_tasks) == case["expected_count"]
    agents_ok = agents_returned == case["expected_agents"]

    status = "✓" if (count_ok and agents_ok) else "✗"
    if count_ok and agents_ok:
        passed += 1

    print(f"{status} '{case['question'][:55]}'")
    print(f"   Got {len(sub_tasks)} sub-task(s): {agents_returned}")
    print(f"   Expected {case['expected_count']} sub-task(s): {case['expected_agents']}")

print(f"\n{passed}/{len(cases)} planner tests passed")