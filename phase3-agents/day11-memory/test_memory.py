# phase3-agents/day11-memory/test_memory.py
from memory import Memory

def test_memory():
    m = Memory()

    print("=== SHORT-TERM ===")
    m.short.store("test_key", "hello world", confidence=0.9)
    result = m.short.retrieve("test_key")
    assert result["value"] == "hello world", "Short-term store/retrieve failed"
    print(f"✓ Short-term: {result}")

    print("\n=== LONG-TERM ===")
    m.long.store("lt_key", "persisted value", confidence=0.8, source="test")
    result = m.long.retrieve("lt_key")
    assert result["value"] == "persisted value", "Long-term store/retrieve failed"
    print(f"✓ Long-term: {result}")

    print("\n=== CONFIDENCE DEGRADATION ===")
    m.long.store("uncertain_fact", "maybe true", confidence=0.7)
    m.long.degrade_confidence("uncertain_fact", amount=0.2)
    result = m.long.retrieve("uncertain_fact")
    assert result["confidence"] < 0.6, "Degradation failed"
    print(f"✓ Degraded: {result}")

    print("\n=== LOW CONFIDENCE QUERY ===")
    low = m.long.low_confidence(threshold=0.6)
    assert any(r["key"] == "uncertain_fact" for r in low)
    print(f"✓ Low-conf entries: {low}")

    print("\n=== UNIFIED FACADE ===")
    m.store("facade_key", "unified value", confidence=0.95)
    result = m.retrieve("facade_key")
    assert result["value"] == "unified value"
    assert result["tier"] == "short"  # short-term wins on fresh store
    print(f"✓ Facade retrieve (short-term): {result}")

    print("\n=== CONTEXT BLOCK ===")
    block = m.context_block()
    print(block)

    print("\n✓ All tests passed.")

if __name__ == "__main__":
    test_memory()