import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chat_exporter.parser import ChatRequest, ChatSession, ResponsePart, ToolCall
from chat_exporter.providers import DiscoveredSession
from chat_exporter.search import search_discovered


def _session(
    session_id: str,
    title: str,
    user: str,
    assistant: str,
    *,
    created_ms: int | None = None,
) -> ChatSession:
    request = ChatRequest(
        request_id="r1",
        user_text=user,
        response_parts=[
            ResponsePart("text", text=assistant),
            ResponsePart(
                "tool_call",
                tool=ToolCall(name="bash", output="reconciliation result: 42"),
            ),
        ],
    )
    return ChatSession(
        session_id=session_id,
        title=title,
        created_ms=created_ms or 1_775_178_266_000,
        model_id="test-model",
        requests=[request],
        provider="copilot-cli",
        workspace="audit-tool",
    )


class SearchTests(unittest.TestCase):
    def test_search_requires_all_terms_and_returns_snippet(self):
        session = _session(
            "session-1",
            "Reconcile store hours",
            "Find the store hours mismatch",
            "The source comparison found a reconciliation issue.",
        )
        items = [DiscoveredSession("copilot-cli", "audit-tool", session_path, session)]
        result = search_discovered(items, "store reconciliation")
        self.assertEqual(result["matching_sessions"], 1)
        hit = result["results"][0]
        self.assertEqual((hit["session_id"], hit["provider"]), ("session-1", "copilot-cli"))
        self.assertIn("reconciliation", hit["snippet"].casefold())

    def test_search_filters_workspace_and_includes_tool_output(self):
        session = _session("session-1", "Investigation", "review it", "No useful text")
        item = DiscoveredSession("copilot-cli", "audit-tool", session_path, session)
        self.assertEqual(
            search_discovered([item], "42", workspace="audit", include_tool_output=True)[
                "matching_sessions"
            ],
            1,
        )
        self.assertEqual(search_discovered([item], "42", workspace="other")["matching_sessions"], 0)

    def test_empty_query_returns_no_results_for_empty_store(self):
        self.assertEqual(search_discovered([], "   ")["results"], [])

    def test_empty_query_returns_most_recent_sessions(self):
        now = datetime.now(timezone.utc)
        older = int((now - timedelta(days=10)).timestamp() * 1000)
        newer = int((now - timedelta(days=1)).timestamp() * 1000)
        items = [
            DiscoveredSession(
                "copilot-cli",
                "audit-tool",
                session_path,
                _session("old", "Older", "old", "old", created_ms=older),
            ),
            DiscoveredSession(
                "copilot-cli",
                "audit-tool",
                session_path,
                _session("new", "Newer", "new", "new", created_ms=newer),
            ),
        ]

        result = search_discovered(items, "", limit=1)

        self.assertEqual(result["search_mode"], "recent")
        self.assertEqual(result["results"][0]["session_id"], "new")

    def test_recent_days_filters_search_and_recent_sort_orders_matches(self):
        now = datetime.now(timezone.utc)
        old = int((now - timedelta(days=20)).timestamp() * 1000)
        recent = int((now - timedelta(days=2)).timestamp() * 1000)
        newest = int((now - timedelta(hours=1)).timestamp() * 1000)
        items = [
            DiscoveredSession(
                "copilot-cli",
                "audit-tool",
                session_path,
                _session("old", "Old", "reconcile", "reconcile", created_ms=old),
            ),
            DiscoveredSession(
                "copilot-cli",
                "audit-tool",
                session_path,
                _session("recent", "Recent", "reconcile", "reconcile", created_ms=recent),
            ),
            DiscoveredSession(
                "copilot-cli",
                "audit-tool",
                session_path,
                _session("newest", "Newest", "reconcile", "reconcile", created_ms=newest),
            ),
        ]

        result = search_discovered(
            items,
            "reconcile",
            recent_days=7,
            sort_by="recent",
        )

        self.assertEqual(result["matching_sessions"], 2)
        self.assertEqual(
            [hit["session_id"] for hit in result["results"]],
            ["newest", "recent"],
        )

    def test_recent_days_must_be_positive(self):
        with self.assertRaises(ValueError):
            search_discovered([], "", recent_days=0)


session_path = Path("session.jsonl")
