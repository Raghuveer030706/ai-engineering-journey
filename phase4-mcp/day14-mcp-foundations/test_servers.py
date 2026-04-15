# phase4-mcp/day14-mcp-foundations/test_servers.py
import asyncio
from client.mcp_client import MCPClient


async def test():
    print("=== Initializing MCP client ===")
    mcp = MCPClient()
    tools = await mcp.initialize()

    print(f"\nTotal tools discovered: {len(tools)}")
    for t in tools:
        print(f"  [{t['server']}] {t['name']} — {t['description'][:60]}")

    print("\n=== Testing local calculator ===")
    result = await mcp.call_tool("calculator", {"expression": "(0.827 - 0.638) / 0.638 * 100"})
    print(f"Result: {result}")
    assert "29" in result, "Calculator result unexpected"
    print("✓ Calculator passed")

    print("\n=== Testing local project_facts ===")
    result = await mcp.call_tool("project_facts", {"key": "phase2_capstone"})
    print(f"Result: {result}")
    assert "0.827" in result, "Project facts result unexpected"
    print("✓ Project facts passed")

    print("\n=== Testing fetch server ===")
    result = await mcp.call_tool("fetch", {"url": "https://example.com"})
    print(f"Result (first 200 chars): {result[:200]}")
    assert len(result) > 10, "Fetch returned empty"
    print("✓ Fetch passed")

    print("\n✓ All server tests passed")


if __name__ == "__main__":
    asyncio.run(test())