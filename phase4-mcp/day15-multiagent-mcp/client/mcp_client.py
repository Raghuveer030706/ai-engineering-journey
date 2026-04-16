# phase4-mcp/day14-mcp-foundations/client/mcp_client.py
import asyncio
import json
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


LOCAL_SERVER_PATH = Path(__file__).parent.parent / "servers" / "local_server.py"

FETCH_SERVER_PARAMS = StdioServerParameters(
    command="uvx",
    args=["mcp-server-fetch"],
    env=None,
)


class MCPClient:
    """
    Manages connections to local and public MCP servers.
    Provides a unified call_tool() interface to the agent.
    Tool routing is automatic — client knows which server owns which tool.
    """

    def __init__(self):
        self._local_tools: list[str] = []
        self._fetch_tools: list[str] = []
        self._all_tools: list[dict] = []

    async def initialize(self):
        """Discover available tools from both servers."""
        self._local_tools, local_schemas = await self._list_tools_local()
        self._fetch_tools, fetch_schemas = await self._list_tools_fetch()
        self._all_tools = local_schemas + fetch_schemas

        print(f"Local server tools: {self._local_tools}")
        print(f"Fetch server tools: {self._fetch_tools}")
        return self._all_tools

    async def _list_tools_local(self):
        params = StdioServerParameters(
            command="python",
            args=[str(LOCAL_SERVER_PATH)],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                names = [t.name for t in result.tools]
                schemas = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema,
                        "server": "local",
                    }
                    for t in result.tools
                ]
                return names, schemas

    async def _list_tools_fetch(self):
        try:
            async with stdio_client(FETCH_SERVER_PARAMS) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    names = [t.name for t in result.tools]
                    schemas = [
                        {
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.inputSchema,
                            "server": "fetch",
                        }
                        for t in result.tools
                    ]
                    return names, schemas
        except Exception as e:
            print(f"Fetch server unavailable: {e}")
            return [], []

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Route tool call to the correct server and return result."""
        if tool_name in self._local_tools:
            return await self._call_local(tool_name, arguments)
        elif tool_name in self._fetch_tools:
            return await self._call_fetch(tool_name, arguments)
        else:
            return f"Error: tool '{tool_name}' not found on any server."

    async def _call_local(self, tool_name: str, arguments: dict) -> str:
        params = StdioServerParameters(
            command="python",
            args=[str(LOCAL_SERVER_PATH)],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result.content[0].text if result.content else "No result"

    async def _call_fetch(self, tool_name: str, arguments: dict) -> str:
        try:
            async with stdio_client(FETCH_SERVER_PARAMS) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    text = result.content[0].text if result.content else "No result"
                    # Truncate long fetch results to avoid context bloat
                    return text[:2000] + "..." if len(text) > 2000 else text
        except Exception as e:
            return f"Fetch server error: {e}"

    def get_tool_descriptions(self) -> str:
        """Returns tool list formatted for agent system prompt."""
        lines = []
        for t in self._all_tools:
            lines.append(f"- {t['name']} [{t['server']}]: {t['description']}")
        return "\n".join(lines)