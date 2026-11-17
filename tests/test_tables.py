import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import polars as pl

from chat_exporter import main as cli
from chat_exporter.parser import (
    ChatRequest,
    ChatSession,
    ResponsePart,
    SubagentCall,
    TodoItem,
    ToolCall,
)
from chat_exporter.tables import TABLES, TableRows, write_parquet

T0 = 1_775_178_266_000


def _session_one() -> ChatSession:
    long_output = "x" * 50
    return ChatSession(
        session_id="s1", title="First", created_ms=T0, model_id="model-a",
        provider="copilot-cli", workspace="demo", source_path="<WORKSPACE>/s1.jsonl",
        cwd="C:/work/demo", git_branch="main", agent_version="1.0",
        requests=[
            ChatRequest(
                request_id="r1", timestamp_ms=T0, user_text="run the tests",
                model_id="model-a", resolved_model="model-b",
                input_tokens=10, completion_tokens=5, context_files=["a.py"],
                response_parts=[
                    ResponsePart("thinking", text="plan"),
                    ResponsePart("text", text="Running."),
                    ResponsePart("tool_call", tool=ToolCall(
                        call_id="c1", name="bash", command="pytest", exit_code=1,
                        is_error=True, output=long_output, arguments='{"a": 1}')),
                    ResponsePart("warning", text="careful"),
                    ResponsePart("text", text="Failed."),
                    ResponsePart("tool_call", tool=ToolCall(
                        call_id="c2", name="task", result_files=["b.py"],
                        todos=[TodoItem("step", "done")],
                        subagent=SubagentCall("look", "find it", "model-c", "found"))),
                    ResponsePart("file_edit", file_path="c.py"),
                ],
                edited_files=["b.py"],
            ),
            ChatRequest(
                request_id="r2", timestamp_ms=T0 + 60_000, is_system_initiated=True,
                system_label="compaction", model_id="model-a",
                response_parts=[ResponsePart("text", text="ok")],
            ),
        ],
    )


def _session_two() -> ChatSession:
    return ChatSession(
        session_id="s2", title="Second", created_ms=0, model_id="model-z",
        provider="copilot-desktop", workspace="other",
        requests=[ChatRequest(request_id="q1", user_text="hi")],
    )


def _rows(**kw) -> TableRows:
    rows = TableRows(**kw)
    rows.add_session(_session_one())
    rows.add_session(_session_two())
    return rows


class RowTests(unittest.TestCase):
    def test_counts_and_keys(self):
        rows = _rows(max_tool_output=0)
        self.assertEqual([len(rows.sessions), len(rows.turns), len(rows.tool_calls)], [2, 3, 2])
        for name, columns in TABLES.items():
            for row in rows.table(name):
                self.assertEqual(list(row), list(columns))

    def test_session_rows(self):
        s1, s2 = _rows().sessions
        self.assertEqual((s1["n_turns"], s1["n_user_turns"]), (2, 1))
        self.assertEqual((s1["n_tool_calls"], s1["n_tool_errors"]), (2, 1))
        self.assertEqual(s1["models"], ["model-b", "model-a"])
        self.assertEqual((s1["input_tokens"], s1["completion_tokens"]), (10, 5))
        self.assertEqual(s1["created_utc"], datetime.fromtimestamp(T0 / 1000, tz=timezone.utc))
        self.assertEqual((s1["last_turn_utc"] - s1["first_turn_utc"]).total_seconds(), 60)
        self.assertIsNone(s2["created_utc"])
        self.assertIsNone(s2["first_turn_utc"])
        self.assertIsNone(s2["input_tokens"])
        self.assertEqual(s2["models"], ["model-z"])
        self.assertEqual(s2["n_tool_calls"], 0)

    def test_turn_rows(self):
        turns = _rows().turns
        first, system, plain = turns
        self.assertEqual([t["turn_index"] for t in turns], [0, 1, 0])
        self.assertEqual(first["assistant_text"], "Running.\n\nFailed.")
        self.assertEqual(first["thinking_text"], "")
        self.assertEqual(first["warnings"], "careful")
        self.assertEqual((first["n_tool_calls"], first["n_tool_errors"], first["n_subagent_calls"]),
                         (2, 1, 1))
        self.assertEqual(first["edited_files"], ["b.py", "c.py"])
        self.assertTrue(system["is_system_initiated"])
        self.assertEqual(system["system_label"], "compaction")
        self.assertIsNone(system["input_tokens"])
        self.assertIsNone(plain["timestamp_utc"])

    def test_thinking_included_on_request(self):
        self.assertEqual(_rows(include_thinking=True).turns[0]["thinking_text"], "plan")

    def test_tool_call_rows(self):
        first, second = _rows(max_tool_output=0).tool_calls
        self.assertEqual((first["call_index"], first["part_index"]), (0, 2))
        self.assertEqual((second["call_index"], second["part_index"]), (1, 5))
        self.assertEqual(first["arguments"], '{"a": 1}')
        self.assertEqual((first["exit_code"], first["is_error"]), (1, True))
        self.assertFalse(first["is_subagent"])
        self.assertTrue(second["is_subagent"])
        self.assertEqual((second["subagent_model"], second["subagent_result"]), ("model-c", "found"))
        self.assertEqual(json.loads(second["todos"]), [{"title": "step", "status": "done"}])
        self.assertEqual(first["todos"], "")
        self.assertEqual(second["result_files"], ["b.py"])

    def test_truncation_and_output_chars(self):
        limited = _rows(max_tool_output=20).tool_calls[0]
        self.assertEqual(limited["output_chars"], 50)
        self.assertTrue(limited["output"].startswith("x" * 20))
        self.assertIn("more characters truncated", limited["output"])
        unlimited = _rows(max_tool_output=0).tool_calls[0]
        self.assertEqual(unlimited["output"], "x" * 50)

    def test_no_tools_keeps_counts(self):
        rows = _rows(include_tools=False)
        self.assertEqual(rows.tool_calls, [])
        self.assertEqual(rows.sessions[0]["n_tool_calls"], 2)
        self.assertEqual(rows.turns[0]["n_tool_calls"], 2)


class WriteTests(unittest.TestCase):
    def test_written_files_and_dtypes(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_parquet(_rows(max_tool_output=0), Path(tmp) / "out")
            self.assertEqual(sorted(written), ["events", "permissions", "sessions", "tool_calls", "turns"])
            sessions = pl.read_parquet(written["sessions"])
            turns = pl.read_parquet(written["turns"])
            calls = pl.read_parquet(written["tool_calls"])
        self.assertEqual((sessions.height, turns.height, calls.height), (2, 3, 2))
        self.assertEqual(sessions.schema["created_utc"], pl.Datetime("us", "UTC"))
        self.assertEqual(sessions.schema["models"], pl.List(pl.String))
        self.assertEqual(sessions.schema["n_turns"], pl.Int64)
        self.assertEqual(turns.schema["is_system_initiated"], pl.Boolean)
        self.assertEqual(turns.schema["reasoning_tokens"], pl.Int64)
        self.assertEqual(calls.schema["exit_code"], pl.Int64)
        self.assertEqual(calls.schema["result_files"], pl.List(pl.String))
        self.assertEqual(turns["reasoning_tokens"].null_count(), 3)

    def test_zero_sessions_keep_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = write_parquet(TableRows(), Path(tmp))
            self.assertEqual(len(list(Path(tmp).glob("*.parquet"))), 5)
            for name, columns in TABLES.items():
                frame = pl.read_parquet(written[name])
                self.assertEqual(frame.height, 0)
                self.assertEqual(frame.columns, list(columns))
            self.assertEqual(pl.read_parquet(written["turns"]).schema["input_tokens"], pl.Int64)
            self.assertEqual(pl.read_parquet(written["events"]).schema["timestamp_utc"],
                             pl.Datetime("us", "UTC"))
            self.assertEqual(pl.read_parquet(written["permissions"]).schema["wait_ms"], pl.Int64)
            self.assertEqual(pl.read_parquet(written["tool_calls"]).schema["context_tokens"],
                             pl.Int64)


def _events_store(tmp: Path) -> Path:
    """A synthetic events.jsonl with three permission requests, one unparsable line
    and one blank line."""
    def ev(kind, second, data, **extra):
        return json.dumps({"type": kind, "data": data, "id": f"e{second}",
                           "parentId": f"e{second - 1}",
                           "timestamp": f"2026-11-13T10:00:{second:02d}.500Z", **extra})
    lines = [
        ev("user.message", 0, {"content": "hi", "interactionId": "i1", "turnId": 0}),
        ev("permission.requested", 1, {"requestId": "R1", "permissionRequest": {
            "kind": "shell", "toolCallId": "c1", "fullCommandText": "ls <WORKSPACE>"}}),
        ev("permission.completed", 3, {"requestId": "R1", "toolCallId": "c1",
                                       "result": {"kind": "approved"},
                                       "decisionSource": "human_response"}),
        "not json at all",
        "",
        ev("permission.requested", 4, {"requestId": "R2", "permissionRequest": {
            "kind": "mcp", "serverName": "demo", "toolName": "query",
            "url": "https://example.test/x", "toolCallId": "c2"}}),
        ev("permission.completed", 5, {"requestId": "R2", "result": {"kind": "denied"},
                                       "decisionSource": "rule"}),
        ev("permission.requested", 6, {"requestId": "R3", "permissionRequest": {
            "kind": "write", "filePath": "C:/work/demo/a.txt"}}),
        ev("assistant.message", 7, {"model": "model-a", "content": "x" * 5000}),
        json.dumps({"type": "tool.execution_start", "data": {"toolCallId": "c9"}}),
    ]
    root = tmp / "store"
    path = root / "session-state" / "sess-1" / "events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return root


class EventTableTests(unittest.TestCase):
    def _rows(self, tmp: str) -> TableRows:
        from chat_exporter.providers import discover
        root = _events_store(Path(tmp))
        rows = TableRows()
        for item in discover({"copilot-cli"}, roots={"copilot-cli": root}):
            rows.add_session(item.session)
        return rows

    def test_events_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs("chat_exporter.tables", level="WARNING") as logs:
                rows = self._rows(tmp)
        self.assertIn("2 blank or unparsable", logs.output[0])
        events = rows.events
        self.assertEqual([e["event_index"] for e in events], [0, 1, 2, 5, 6, 7, 8, 9])
        self.assertTrue(all(list(e) == list(TABLES["events"]) for e in events))
        user, request = events[0], events[1]
        self.assertEqual((user["type"], user["turn_id"], user["interaction_id"]),
                         ("user.message", "0", "i1"))
        self.assertEqual(user["timestamp_utc"],
                         datetime(2026, 11, 13, 10, 0, 0, 500000, tzinfo=timezone.utc))
        self.assertEqual((request["event_id"], request["parent_id"]), ("e1", "e0"))
        self.assertEqual(request["tool_call_id"], "c1")
        self.assertEqual(request["data_keys"], ["permissionRequest", "requestId"])
        self.assertEqual(json.loads(request["data_json"])["requestId"], "R1")
        long = events[6]
        self.assertEqual((long["model"], len(long["data_json"])), ("model-a", 4000))
        self.assertGreater(long["data_chars"], 5000)
        last = events[-1]
        self.assertIsNone(last["timestamp_utc"])
        self.assertEqual(last["tool_call_id"], "c9")

    def test_permission_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs("chat_exporter.tables", level="WARNING"):
                rows = self._rows(tmp)
        approved, denied, never = rows.permissions
        self.assertTrue(all(list(p) == list(TABLES["permissions"]) for p in rows.permissions))
        self.assertEqual((approved["request_id"], approved["tool_call_id"], approved["kind"]),
                         ("R1", "c1", "shell"))
        self.assertEqual(approved["command"], "ls <WORKSPACE>")
        self.assertEqual((approved["result_kind"], approved["decision_source"]),
                         ("approved", "human_response"))
        self.assertEqual(approved["wait_ms"], 2000)
        self.assertEqual(json.loads(approved["result_json"]), {"kind": "approved"})
        self.assertEqual((denied["tool_name"], denied["url"]), ("demo/query", "https://example.test/x"))
        self.assertEqual((denied["result_kind"], denied["decision_source"]), ("denied", "rule"))
        self.assertEqual(denied["wait_ms"], 1000)
        self.assertEqual(never["path"], "C:/work/demo/a.txt")
        self.assertIsNone(never["completed_utc"])
        self.assertIsNone(never["wait_ms"])
        self.assertEqual((never["result_kind"], never["decision_source"], never["result_json"]),
                         ("", "", ""))

    def test_other_providers_and_unreadable_files_add_no_rows(self):
        rows = TableRows()
        other = _session_one()
        other.provider, other.source_path = "opencode", "<WORKSPACE>/s1.jsonl"
        rows.add_session(other)
        gone = _session_two()
        gone.source_path = "<WORKSPACE>/missing/events.jsonl"
        with self.assertLogs("chat_exporter.tables", level="WARNING"):
            rows.add_session(gone)
        self.assertEqual((rows.events, rows.permissions), ([], []))
        self.assertEqual(len(rows.sessions), 2)

    def test_write_parquet_with_event_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs("chat_exporter.tables", level="WARNING"):
                rows = self._rows(tmp)
            written = write_parquet(rows, Path(tmp) / "out")
            events = pl.read_parquet(written["events"])
            permissions = pl.read_parquet(written["permissions"])
        self.assertEqual((events.height, permissions.height), (8, 3))
        self.assertEqual(events.schema["timestamp_utc"], pl.Datetime("us", "UTC"))
        self.assertEqual(events.schema["data_keys"], pl.List(pl.String))
        self.assertEqual(events.schema["data_chars"], pl.Int64)
        self.assertEqual(permissions.schema["wait_ms"], pl.Int64)
        self.assertEqual(permissions.schema["completed_utc"], pl.Datetime("us", "UTC"))
        self.assertEqual(permissions["completed_utc"].null_count(), 1)

    def test_tool_call_failure_columns(self):
        rows = _rows()
        call = rows.tool_calls[0]
        keys = list(call)
        self.assertEqual(keys[keys.index("is_complete") + 1:keys.index("is_complete") + 7],
                         ["error_code", "error_message", "failure_category", "failure_stage",
                          "context_tokens", "context_limit"])
        self.assertEqual((call["error_code"], call["context_tokens"]), ("", None))


class CliTests(unittest.TestCase):
    def _store(self, tmp: Path) -> Path:
        state = tmp / "store" / "session-state" / "sess-1"
        state.mkdir(parents=True)
        events = [
            {"type": "session.start", "timestamp": "2026-08-27T01:44:55.640Z", "data": {
                "sessionId": "sess-1", "copilotVersion": "1.0.80",
                "startTime": "2026-08-27T01:44:55.640Z", "context": {"cwd": "/work/demo"}}},
            {"type": "user.message", "timestamp": "2026-08-27T01:45:00Z",
             "data": {"content": "hello", "interactionId": "i1"}},
            {"type": "assistant.message", "timestamp": "2026-08-27T01:45:02Z",
             "data": {"model": "model-a", "content": "hi"}},
        ]
        (state / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        return tmp / "store"

    def _run(self, *argv: str) -> str:
        buffer = io.StringIO()
        with mock.patch.object(sys, "argv", ["chat-exporter", *argv]), \
                contextlib.redirect_stdout(buffer):
            cli.main()
        return buffer.getvalue()

    def test_cli_parquet_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = self._store(tmp_path)
            out = tmp_path / "out"
            text = self._run("--format", "parquet", "--provider", "copilot-cli",
                             "--root", f"copilot-cli={root}", "-o", str(out))
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["events.parquet", "permissions.parquet", "sessions.parquet",
                              "tool_calls.parquet", "turns.parquet"])
            self.assertIn("sessions: 1, turns: 1, tool_calls: 0, events: 3, permissions: 0", text)
            self.assertIn("1 session(s) exported", text)
            turns = pl.read_parquet(out / "turns.parquet")
            self.assertEqual(turns["user_text"].to_list(), ["hello"])
            self.assertEqual(turns["assistant_text"].to_list(), ["hi"])

    def test_cli_parquet_min_messages_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = self._store(tmp_path)
            out = tmp_path / "out"
            text = self._run("--format", "parquet", "--provider", "copilot-cli",
                             "--root", f"copilot-cli={root}", "-o", str(out),
                             "--min-messages", "5")
            self.assertIn("0 session(s) exported, 1 skipped", text)
            self.assertEqual(pl.read_parquet(out / "sessions.parquet").height, 0)


if __name__ == "__main__":
    unittest.main()


class ScalarAndOutcomeTests(unittest.TestCase):
    def test_scalars_and_nested_json_always_parse(self):
        from chat_exporter.tables import NESTED_KEYS, nested_json, scalars_json
        data = {"s": "ok", "n": 3, "f": 1.5, "b": False, "none": None, "long": "y" * 5000,
                "list": [1], "obj": {"a": 1}, "bad": float("nan"),
                "result": {"kind": "approved", "deep": {"x": 1}, "text": "z" * 500},
                "error": "e" * 400, "assistedApprovalAssessment": {"risk": "low", "l": [1]},
                "modelMetrics": {"m": {"n": 1}}}
        flat = json.loads(scalars_json(data))
        self.assertEqual(flat, {"s": "ok", "n": 3, "f": 1.5, "b": False})
        nested = json.loads(nested_json(data))
        self.assertEqual(list(nested), list(NESTED_KEYS))
        self.assertEqual(nested["result"], {"kind": "approved", "text": "z" * 300})
        self.assertEqual(nested["error"], "e" * 300)
        self.assertEqual(nested["assistedApprovalAssessment"], {"risk": "low"})
        self.assertEqual(json.loads(nested_json({"result": 5, "error": None})), {})
        self.assertEqual(json.loads(scalars_json({})), {})

    def test_event_row_tolerates_odd_data(self):
        from chat_exporter.tables import event_row
        session = _session_one()
        for odd in (None, "text", [1]):
            row = event_row(session, 0, {"type": "x", "data": odd})
            self.assertEqual((row["scalars_json"], row["nested_json"]), ("{}", "{}"))
        row = event_row(session, 0, {"type": "x", "data": {"result": ["x"], "error": 5}})
        self.assertEqual(json.loads(row["scalars_json"]), {"error": 5})
        self.assertEqual(row["nested_json"], "{}")

    def test_permission_outcome_mapping(self):
        from chat_exporter.tables import permission_rows
        gone = "denied-no-approval-rule-and-could-not-request-from-user"
        kinds = ["approved", "approved-for-location", "approved-read-only-for-session",
                 "denied-interactively-by-user", gone, "cancelled", "weird", ""]
        events = []
        for i, kind in enumerate(kinds):
            rid = f"R{i}"
            events.append({"type": "permission.requested", "data": {
                "requestId": rid, "agentMode": "autopilot", "permissionMode": "default",
                "recoveryEpisodeId": "ep", "permissionRequest": {"kind": "shell"},
                "assistedApprovalAssessment": {"note": "q" * 2000}}})
            events.append({"type": "permission.completed", "data": {
                "requestId": rid, "result": {"kind": kind}}})
        events.append({"type": "permission.requested", "data": {"requestId": "open"}})
        rows = permission_rows(_session_one(), events)
        self.assertEqual([r["outcome"] for r in rows], [
            "approved", "approved_saved", "approved_saved", "denied_by_user",
            "denied_unattended", "cancelled", "other", "pending", "pending"])
        self.assertEqual([r["approved"] for r in rows],
                         [True, True, True, False, False, False, False, False, False])
        first, last = rows[0], rows[-1]
        self.assertEqual((first["agent_mode"], first["permission_mode"],
                          first["recovery_episode_id"]), ("autopilot", "default", "ep"))
        self.assertEqual(len(first["assisted_assessment_json"]), 1000)
        self.assertEqual((last["agent_mode"], last["assisted_assessment_json"]), ("", ""))
        self.assertTrue(all(list(r) == list(TABLES["permissions"]) for r in rows))
