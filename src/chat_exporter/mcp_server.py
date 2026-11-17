from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from chat_exporter.search import (
    DEFAULT_LIMIT,
    DEFAULT_MAX_CHARS,
    MAX_CHARS,
    read_session,
    search_history,
)

mcp = FastMCP(
    "chat-history",
    instructions=(
        "Search the local history of AI coding chats captured by Copilot Chat, "
        "Copilot CLI, Claude Code, Codex, AGY, and OpenCode. Use search_chat_history "
        "first with distinctive words from the user's question, then use "
        "read_chat_session to inspect a matching transcript. Searches read the "
        "current local stores directly and are read-only. Results are bounded; "
        "for cross-session review, use read_chat_session with condensed=True to "
        "omit tool-call noise and keep only user prompts plus assistant prose. "
        "filter by providers or workspace when the query is broad. To browse "
        "recent chats, omit the query and set limit; use recent_days to bound "
        "the age. Do not treat an empty result as proof that no older conversation "
        "exists because a client may store history in an unavailable or custom "
        "location."
    ),
)


@mcp.tool()
def search_chat_history(
    query: str = "",
    providers: list[str] | None = None,
    workspace: str | None = None,
    limit: int = DEFAULT_LIMIT,
    include_tool_output: bool = False,
    recent_days: int | None = None,
    sort_by: Literal["relevance", "recent"] = "relevance",
) -> dict[str, Any]:
    """Search past local coding chats or browse recent sessions.

    Provider names are copilot, copilot-cli, agy, claude, codex, and opencode.
    Omit providers to search all supported stores. Set include_tool_output=True
    when command results are part of the question; it is false by default so a
    search does not match its own MCP request or other tool-call noise.
    Set recent_days to search only sessions created in the last N days. Omit
    query to browse the most recent sessions; limit controls how many are
    returned. Set sort_by="recent" to order matching results by recency instead
    of relevance.
    """
    return search_history(
        query,
        providers=providers,
        workspace=workspace,
        limit=limit,
        include_tool_output=include_tool_output,
        recent_days=recent_days,
        sort_by=sort_by,
    )


@mcp.tool()
def read_chat_session(
    provider: str,
    session_id: str,
    workspace: str | None = None,
    include_tools: bool = True,
    max_tool_output: int = 2_000,
    max_chars: int = DEFAULT_MAX_CHARS,
    condensed: bool = False,
    max_assistant_chars: int = 2_000,
) -> dict[str, Any]:
    """Read one matching chat transcript after search_chat_history identifies it.

    Set condensed=True to return only user prompts and bounded assistant prose,
    without tool calls, metrics, file lists, or context metadata.
    """
    if max_chars > MAX_CHARS:
        max_chars = MAX_CHARS
    if max_chars <= 0:
        max_chars = DEFAULT_MAX_CHARS
    return read_session(
        provider,
        session_id,
        workspace=workspace,
        include_tools=include_tools,
        max_tool_output=max_tool_output,
        max_chars=max_chars,
        condensed=condensed,
        max_assistant_chars=max_assistant_chars,
    )


def main() -> None:
    """Run the chat-history MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
