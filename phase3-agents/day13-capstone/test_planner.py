# phase3-agents/day13-capstone/test_planner.py
from planner import plan

cases = [
    {
        "question": "What is HyDE?",
        "expected_agents": ["rag"],
        "expected_count": 1,
    },
    {
        "question": "What is semantic chunking and how many chunks did Day 3 produce?",
        "expected_agents": ["rag"],
        "expected_count": 1,
    },
    {
        "question": "Calculate 0.827 minus 0.638 and store the result in memory as 'phase2_delta'",
        "expected_agents": ["math", "memory"],
        "expected_count": 2,
    },
    {
        "question": "What was the Day 9 RAGAS score, multiply it by 100, and store as 'day9_pct'",
        "expected_agents": ["rag", "math", "memory"],
        "expected_count": 3,
    },
]

passed = 0
for case in cases:
    sub_tasks = plan(case["question"])
    agents_returned = [st["agent"] for st in sub_tasks]
    count_ok = len(sub_tasks) == case["expected_count"]
    agents_ok = agents_returned == case["expected_agents"]  # exact order match

    status = "✓" if (count_ok and agents_ok) else "✗"
    if count_ok and agents_ok:
        passed += 1

    print(f"{status} '{case['question'][:50]}'")
    print(f"   Got {len(sub_tasks)} sub-task(s): {agents_returned}")
    print(f"   Expected {case['expected_count']} sub-task(s): {case['expected_agents']}")

print(f"\n{passed}/{len(cases)} planner tests passed")