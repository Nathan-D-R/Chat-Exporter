import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from chat_exporter.parser import parse_session
from chat_exporter.providers import (
    discover,
    discover_copilot_cli,
    discover_opencode,
    parse_agy,
    parse_claude,
    parse_codex,
)


def _field(number: int, value: bytes) -> bytes:
    def varint(n: int) -> bytes:
        result = bytearray()
        while n > 127:
            result.append((n & 127) | 128)
            n >>= 7
        result.append(n)
        return bytes(result)
    return varint(number << 3 | 2) + varint(len(value)) + value


class ProviderTests(unittest.TestCase):
    def test_claude(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            rows = [
                {"type": "user", "uuid": "u1", "timestamp": "2026-01-01T00:00:00Z",
                 "cwd": "/work/demo", "gitBranch": "main", "version": "2.1.0",
                 "message": {"role": "user", "content": "hello"}},
                {"type": "assistant", "message": {"role": "assistant", "model": "claude-test",
                 "content": [{"type": "text", "text": "hi"},
                             {"type": "tool_use", "name": "Read", "input": {"path": "x"}}]}},
            ]
            path.write_text("\n".join(json.dumps(x) for x in rows))
            session = parse_claude(path)
            self.assertEqual((session.workspace, session.model_id), ("demo", "claude-test"))
            self.assertEqual(session.requests[0].assistant_text, "hi")
            self.assertEqual((session.git_branch, session.agent_version), ("main", "2.1.0"))

    def test_claude_tool_results_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.jsonl"
            rows = [
                {"type": "user", "uuid": "u1", "timestamp": "2026-01-01T00:00:00Z",
                 "cwd": "/work/demo", "message": {"role": "user", "content": "run it"}},
                # One API response is written as one record per content block,
                # each repeating that call's usage under the same message id.
                # Its 10 output tokens must be counted once, not once per block.
                {"type": "assistant", "timestamp": "2026-01-01T00:00:02Z",
                 "message": {"role": "assistant", "id": "msg_a", "model": "claude-test",
                             "stop_reason": "tool_use",
                             "usage": {"output_tokens": 10, "input_tokens": 5,
                                       "cache_read_input_tokens": 900,
                                       "cache_creation_input_tokens": 40},
                             "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                          "input": {"command": "ls", "description": "list"}}]}},
                {"type": "assistant", "timestamp": "2026-01-01T00:00:02Z",
                 "message": {"role": "assistant", "id": "msg_a", "model": "claude-test",
                             "stop_reason": "tool_use",
                             "usage": {"output_tokens": 10, "input_tokens": 5,
                                       "cache_read_input_tokens": 900,
                                       "cache_creation_input_tokens": 40},
                             "content": [{"type": "tool_use", "id": "t2", "name": "Edit",
                                          "input": {"file_path": "/work/demo/a.py"}}]}},
                # Tool results arrive as user messages and must not open a new turn.
                {"type": "user", "timestamp": "2026-01-01T00:00:03Z",
                 "toolUseResult": {"stdout": "a.py\n", "stderr": "", "interrupted": False},
                 "message": {"role": "user", "content": [
                     {"type": "tool_result", "tool_use_id": "t1", "content": "a.py\n"}]}},
                {"type": "user", "timestamp": "2026-01-01T00:00:04Z",
                 "toolUseResult": {"type": "update", "filePath": "/work/demo/a.py"},
                 "message": {"role": "user", "content": [
                     {"type": "tool_result", "tool_use_id": "t2", "is_error": True,
                      "content": [{"type": "text", "text": "no match"}]}]}},
                # A second call in the same turn. Streaming can revise a count
                # upward under one id; the last record carries the final value.
                {"type": "assistant", "timestamp": "2026-01-01T00:00:05Z",
                 "message": {"role": "assistant", "id": "msg_b", "model": "claude-test",
                             "stop_reason": "end_turn",
                             "usage": {"output_tokens": 4, "input_tokens": 6,
                                       "cache_read_input_tokens": 1200},
                             "content": [{"type": "text", "text": "done"}]}},
                {"type": "assistant", "timestamp": "2026-01-01T00:00:05Z",
                 "message": {"role": "assistant", "id": "msg_b", "model": "claude-test",
                             "stop_reason": "end_turn",
                             "usage": {"output_tokens": 9, "input_tokens": 6,
                                       "cache_read_input_tokens": 1200},
                             "content": [{"type": "text", "text": "done"}]}},
            ]
            path.write_text("\n".join(json.dumps(x) for x in rows))
            session = parse_claude(path)
            self.assertEqual(len(session.requests), 1)
            req = session.requests[0]
            bash, edit = req.tool_calls
            self.assertEqual((bash.output, bash.command, bash.message), ("a.py\n", "ls", "list"))
            self.assertEqual((edit.output, edit.is_error), ("no match", True))
            self.assertEqual(req.edited_files, ["/work/demo/a.py"])
            # 10 for msg_a counted once despite two records, plus msg_b's
            # revised 9 rather than its superseded 4. Context is the last call's.
            self.assertEqual((req.completion_tokens, req.input_tokens), (19, 6))
            self.assertEqual((req.cache_read_tokens, req.cache_write_tokens), (1200, 40))
            self.assertEqual((req.stop_reason, req.total_elapsed_ms), ("end_turn", 5000))

    def test_claude_subagent_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent-a1.jsonl").write_text("\n".join(json.dumps(x) for x in [
                {"type": "assistant", "isSidechain": True,
                 "message": {"role": "assistant", "content": [{"type": "text", "text": "early"}]}},
                {"type": "assistant", "isSidechain": True,
                 "message": {"role": "assistant", "content": [{"type": "text", "text": "report"}]}},
            ]))
            path = root / "session.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in [
                {"type": "user", "uuid": "u1", "cwd": "/work/demo",
                 "message": {"role": "user", "content": "delegate"}},
                {"type": "assistant", "message": {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Agent",
                     "input": {"description": "dig"}}]}},
                {"type": "user", "toolUseResult": {
                    "agentId": "a1", "description": "dig", "prompt": "go look",
                    "resolvedModel": "claude-sub"},
                 "message": {"role": "user", "content": [
                     {"type": "tool_result", "tool_use_id": "t1", "content": "launched"}]}},
            ]))
            # The agent transcript is all sidechain, so it is not a session of its own.
            self.assertIsNone(parse_claude(root / "agent-a1.jsonl"))
            sub = parse_claude(path).requests[0].tool_calls[0].subagent
            self.assertEqual((sub.description, sub.prompt), ("dig", "go look"))
            self.assertEqual((sub.model_name, sub.result), ("claude-sub", "report"))

    def test_codex(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-id.jsonl"
            rows = [
                {"type": "session_meta", "payload": {"id": "s1", "cwd": "/work/demo",
                                                       "timestamp": "2026-01-01T00:00:00Z"}},
                {"timestamp": "2026-01-01T00:00:01Z", "type": "response_item",
                 "payload": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": "hello"}]}},
                {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "hi"}]}},
            ]
            path.write_text("\n".join(json.dumps(x) for x in rows))
            session = parse_codex(path)
            self.assertEqual((session.session_id, session.workspace), ("s1", "demo"))
            self.assertEqual(session.requests[0].assistant_text, "hi")

    def test_codex_command_timing_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-id.jsonl"
            rows = [
                {"type": "session_meta", "payload": {"id": "s1", "cwd": "/work/demo",
                                                     "cli_version": "0.1.0",
                                                     "git": {"branch": "main"},
                                                     "timestamp": "2026-01-01T00:00:00Z"}},
                {"type": "turn_context", "payload": {"turn_id": "t", "model": "gpt-test"}},
                {"type": "response_item", "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hello"}]}},
                {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec",
                                                      "call_id": "c1", "input": "ls"}},
                # The event carries an exec- id, so it joins to the call by position.
                {"type": "event_msg", "payload": {
                    "type": "item_completed", "started_at_ms": 1000, "completed_at_ms": 1750,
                    "item": {"type": "CommandExecution", "id": "exec-1", "command": ["sh", "-c", "ls"],
                             "cwd": "/work/demo", "exit_code": 2, "status": "failed",
                             "aggregated_output": "boom"}}},
                {"type": "response_item", "payload": {"type": "custom_tool_call_output",
                                                      "call_id": "c1", "output": "ignored"}},
                {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                    "type": "FileChange", "changes": {"/work/demo/a.py": {"type": "update"}}}}},
                {"type": "event_msg", "payload": {"type": "token_count", "info": {
                    "last_token_usage": {"input_tokens": 100, "cached_input_tokens": 80,
                                         "output_tokens": 7, "reasoning_output_tokens": 3}}}},
                {"type": "event_msg", "payload": {"type": "task_complete", "duration_ms": 4200,
                                                  "time_to_first_token_ms": 300}},
            ]
            path.write_text("\n".join(json.dumps(x) for x in rows))
            req = parse_codex(path).requests[0]
            tool = req.tool_calls[0]
            self.assertEqual((tool.command, tool.exit_code, tool.cwd),
                             ("sh -c ls", 2, "/work/demo"))
            # The event's own output wins over the generic call output.
            self.assertEqual((tool.output, tool.is_error, tool.duration_ms), ("boom", True, 750))
            self.assertEqual((req.model_id, req.edited_files), ("gpt-test", ["/work/demo/a.py"]))
            self.assertEqual((req.completion_tokens, req.input_tokens), (7, 100))
            self.assertEqual((req.cache_read_tokens, req.reasoning_tokens), (80, 3))
            self.assertEqual((req.total_elapsed_ms, req.first_progress_ms), (4200, 300))
            session = parse_codex(path)
            self.assertEqual((session.git_branch, session.agent_version), ("main", "0.1.0"))

    def test_codex_file_change_list_shape(self):
        """`changes` may be a list of {path, kind} rather than a path-keyed object."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-id.jsonl"
            rows = [
                {"type": "session_meta", "payload": {"id": "s1", "cwd": "/work/demo",
                                                     "timestamp": "2026-01-01T00:00:00Z"}},
                {"type": "response_item", "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hello"}]}},
                {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                    "type": "FileChange", "changes": [
                        {"path": "/work/demo/a.py", "kind": "update"},
                        {"path": "/work/demo/b.py", "kind": "add"},
                    ]}}},
            ]
            path.write_text("\n".join(json.dumps(x) for x in rows))
            session = parse_codex(path)
            self.assertEqual(session.requests[0].edited_files,
                             ["/work/demo/a.py", "/work/demo/b.py"])

    def test_codex_command_join_does_not_steal(self):
        """A CommandExecution must not overwrite an unrelated earlier call.

        The second turn's event belongs to a shell invocation this parser does
        not record as a function_call, so it must attach to nothing rather than
        to the still-pending call from the first turn.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-id.jsonl"
            rows = [
                {"type": "session_meta", "payload": {"id": "s1", "cwd": "/work/demo",
                                                     "timestamp": "2026-01-01T00:00:00Z"}},
                {"type": "response_item", "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "first"}]}},
                # A non-shell call, so no CommandExecution ever arrives for it.
                {"type": "response_item", "payload": {"type": "function_call", "name": "web_search",
                                                      "call_id": "c1", "arguments": "{}"}},
                {"type": "response_item", "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "second"}]}},
                {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                    "type": "CommandExecution", "id": "exec-9", "command": ["sh", "-c", "rm -rf x"],
                    "cwd": "/work/demo", "exit_code": 0, "aggregated_output": "gone"}}},
            ]
            path.write_text("\n".join(json.dumps(x) for x in rows))
            session = parse_codex(path)
            search = session.requests[0].tool_calls[0]
            self.assertEqual((search.name, search.command, search.output), ("web_search", "", ""))
            self.assertIsNone(search.exit_code)

    def _copilot_session(self, tmp, rounds, extra_metadata=None, completion=None):
        """Write a minimal copilot chat store shaped like the real event log."""
        path = Path(tmp) / "session.jsonl"
        request = {
            "requestId": "r1", "timestamp": 1775178266074, "modelId": "copilot/auto",
            "message": {"text": "do it"},
            "response": [{"kind": "toolInvocationSerialized", "toolId": "copilot_findFiles",
                          "toolCallId": "call_1", "isComplete": True,
                          "invocationMessage": {"value": "Searching"}}],
            "result": {"metadata": dict({
                "resolvedModel": "claude-sonnet-4-6",
                "promptTokens": 21413,
                "outputTokens": 71,
                "toolCallRounds": rounds,
            }, **(extra_metadata or {}))},
        }
        if completion is not None:
            request["completionTokens"] = completion
        rows = [
            {"kind": 0, "v": {"sessionId": "s1", "creationDate": 1775178266000, "requests": []}},
            {"kind": 2, "k": ["requests"], "v": [request]},
        ]
        path.write_text("\n".join(json.dumps(x) for x in rows))
        return parse_session(path)

    def test_copilot_metadata_usage_and_model(self):
        """promptTokens, resolvedModel, and the single-round token fallback."""
        one_round = [{"id": "call_1__vscode-1", "toolCalls": [
            {"id": "call_1__vscode-1", "name": "file_search", "arguments": '{"query": "*.py"}'}]}]
        with tempfile.TemporaryDirectory() as tmp:
            session = self._copilot_session(tmp, one_round)
            req = session.requests[0]
            self.assertEqual(req.resolved_model, "claude-sonnet-4-6")
            self.assertEqual(req.input_tokens, 21413)
            # One round, so metadata.outputTokens is the whole turn's output.
            self.assertEqual(req.completion_tokens, 71)
            self.assertEqual(session.peak_context_tokens, 21413)

    def test_copilot_multi_round_does_not_use_output_tokens(self):
        """outputTokens is only the last round, so it must not stand in."""
        two_rounds = [
            {"id": "call_1__vscode-1", "toolCalls": [
                {"id": "call_1__vscode-1", "name": "file_search", "arguments": "{}"}]},
            {"id": "call_2__vscode-2", "toolCalls": []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            session = self._copilot_session(tmp, two_rounds)
            self.assertIsNone(session.requests[0].completion_tokens)
        # An explicit completionTokens is the turn total and always wins.
        with tempfile.TemporaryDirectory() as tmp:
            session = self._copilot_session(tmp, two_rounds, completion=3067)
            self.assertEqual(session.requests[0].completion_tokens, 3067)

    def test_copilot_tool_results_from_metadata(self):
        """Output kept only in metadata.toolCallResults is recovered."""
        rounds = [{"id": "call_1__vscode-1", "toolCalls": [
            {"id": "call_1__vscode-1", "name": "file_search", "arguments": "{}"}]}]
        results = {"toolCallResults": {"call_1__vscode-1": {"content": [{"value": {"node": {
            "children": [
                {"text": "736 total results"},
                {"children": [{"text": "/work/demo/a.py"}]},
            ]}}}]}}}
        with tempfile.TemporaryDirectory() as tmp:
            session = self._copilot_session(tmp, rounds, extra_metadata=results)
            tool = session.requests[0].tool_calls[0]
            self.assertEqual(tool.name, "file_search")
            self.assertEqual(tool.output, "736 total results\n/work/demo/a.py")

    def _copilot_cli_store(self, tmp, events, usage_rows=()):
        """Build a ~/.copilot store: an event log plus the sqlite usage index."""
        root = Path(tmp)
        state = root / "session-state" / "sess-1"
        state.mkdir(parents=True)
        (state / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        # A sibling directory holding only rewind snapshots must be skipped.
        (root / "session-state" / "sess-empty" / "rewind-file-snapshots").mkdir(parents=True)
        db = sqlite3.connect(root / "session-store.db")
        db.execute(
            "CREATE TABLE assistant_usage_events (id INTEGER PRIMARY KEY, session_id TEXT,"
            " turn_index INTEGER, model TEXT, input_tokens INT, output_tokens INT,"
            " cache_read_tokens INT, cache_write_tokens INT, reasoning_tokens INT,"
            " duration_ms REAL, time_to_first_token_ms REAL, finish_reason TEXT)"
        )
        db.executemany(
            "INSERT INTO assistant_usage_events (session_id, turn_index, model, input_tokens,"
            " output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,"
            " duration_ms, time_to_first_token_ms, finish_reason)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)", usage_rows)
        db.commit()
        db.close()
        return list(discover_copilot_cli(root))

    def test_copilot_cli_events_and_usage(self):
        events = [
            {"type": "session.start", "timestamp": "2026-08-27T01:44:55.640Z", "data": {
                "sessionId": "sess-1", "copilotVersion": "1.0.80",
                "startTime": "2026-08-27T01:44:55.640Z", "context": {"cwd": "/work/demo"}}},
            {"type": "session.model_change", "data": {"newModel": "auto"}},
            {"type": "session.auto_mode_resolved", "data": {"chosenModel": "gpt-5.6-luna"}},
            {"type": "user.message", "timestamp": "2026-08-27T01:45:00Z",
             "data": {"content": "list it", "interactionId": "i1"}},
            {"type": "tool.execution_start", "data": {
                "toolCallId": "c1", "toolName": "bash",
                "arguments": {"command": "ls -a", "description": "list files"}}},
            {"type": "tool.execution_complete", "data": {
                "toolCallId": "c1", "success": True, "result": {"content": "a.py\nb.py"}}},
            {"type": "assistant.message", "timestamp": "2026-08-27T01:45:02Z", "data": {
                "model": "gpt-5.6-luna", "content": "Two files.", "reasoningText": "checking"}},
        ]
        # Two calls in one turn. Output, cache writes and reasoning accumulate;
        # input and cache reads are one call's context, so they peak instead.
        rows = [("sess-1", 0, "gpt-5.6-luna", 15354, 6, 0, 15351, 0, 1773.0, 1676.474934, "tool_calls"),
                ("sess-1", 0, "gpt-5.6-luna", 82, 300, 15351, 0, 34, 900.4, 500.0, "stop")]
        with tempfile.TemporaryDirectory() as tmp:
            found = self._copilot_cli_store(tmp, events, rows)
            self.assertEqual(len(found), 1)          # the snapshot-only dir is skipped
            session = found[0].session
            self.assertEqual((session.provider, session.workspace), ("copilot-cli", "demo"))
            self.assertEqual(session.agent_version, "1.0.80")
            req = session.requests[0]
            self.assertEqual((req.model_id, req.resolved_model), ("gpt-5.6-luna", "gpt-5.6-luna"))
            # 15354 is the larger context, not 15354 + 82.
            self.assertEqual((req.input_tokens, req.completion_tokens), (15354, 306))
            self.assertEqual((req.cache_read_tokens, req.reasoning_tokens), (15351, 34))
            self.assertEqual(req.cache_write_tokens, 15351)
            # Peak context is one call's input plus its cache reads.
            self.assertEqual(session.peak_context_tokens, 15354 + 15351)
            # Floats round rather than being discarded, and the turn ends on "stop".
            self.assertEqual((req.total_elapsed_ms, req.first_progress_ms), (2673, 1676))
            self.assertEqual(req.stop_reason, "stop")
            bash = req.tool_calls[0]
            self.assertEqual((bash.name, bash.command, bash.message), ("bash", "ls -a", "list files"))
            self.assertEqual((bash.output, bash.is_error), ("a.py\nb.py", False))
            self.assertEqual([p.kind for p in req.response_parts],
                             ["tool_call", "thinking", "text"])

    def test_copilot_cli_older_build_and_abort(self):
        """Builds before usage tracking still yield reasoning, output and aborts."""
        events = [
            {"type": "session.start", "timestamp": "2026-04-18T23:03:00Z", "data": {
                "sessionId": "sess-1", "copilotVersion": "0.0.410",
                "context": {"cwd": "/work/demo"}}},
            {"type": "user.message", "timestamp": "2026-04-18T23:06:50Z",
             "data": {"content": "debug it"}},
            {"type": "assistant.message", "data": {
                "content": "Looking.", "reasoningText": "check the logs", "outputTokens": 42}},
            {"type": "tool.execution_start", "data": {
                "toolCallId": "c1", "toolName": "view", "arguments": {"path": "/var/log"}}},
            {"type": "tool.execution_complete", "data": {
                "toolCallId": "c1", "success": False, "result": {"content": "denied"}}},
            {"type": "abort", "data": {"reason": "user_initiated"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            session = self._copilot_cli_store(tmp, events)[0].session
            req = session.requests[0]
            # No sqlite row, so the event's own count is used instead.
            self.assertEqual(req.completion_tokens, 42)
            self.assertTrue(req.is_incomplete)
            self.assertEqual(req.error_code, "aborted")
            view = req.tool_calls[0]
            self.assertEqual((view.output, view.is_error), ("denied", True))
            self.assertIn('"path": "/work/demo"'.replace("/work/demo", "/var/log"), view.arguments)

    def test_discover_honours_provider_roots(self):
        """A store outside the current home is reachable via `roots`."""
        events = [
            {"type": "session.start", "timestamp": "2026-08-27T01:44:55Z", "data": {
                "sessionId": "sess-1", "copilotVersion": "1.0.80",
                "context": {"cwd": "/work/demo"}}},
            {"type": "user.message", "timestamp": "2026-08-27T01:45:00Z",
             "data": {"content": "hello"}},
            {"type": "assistant.message", "data": {"model": "gpt-5.6-luna", "content": "hi"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._copilot_cli_store(tmp, events)
            # Nothing is found at the default location for this temp store...
            relocated = discover({"copilot-cli"}, roots={"copilot-cli": Path(tmp)})
            self.assertEqual(len(relocated), 1)
            session = relocated[0].session
            self.assertEqual((session.provider, session.workspace), ("copilot-cli", "demo"))
            self.assertEqual(session.requests[0].assistant_text, "hi")

    def test_agy_unknown_protobuf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agy-id.db"
            db = sqlite3.connect(path)
            db.execute("create table steps (idx integer, step_type integer, step_payload blob)")
            user_payload = _field(19, _field(2, b"hello"))
            assistant_payload = _field(114, _field(1, b"hi"))
            db.executemany("insert into steps values (?, ?, ?)", [(0, 14, user_payload), (1, 101, assistant_payload)])
            db.commit()
            db.close()
            session = parse_agy(path, {"title": "Test", "workspace_uris": '["file:///work/demo"]'})
            self.assertEqual(session.workspace, "demo")
            self.assertEqual(session.requests[0].assistant_text, "hi")

    def test_opencode_json_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "storage/session/proj").mkdir(parents=True)
            (root / "storage/message/s1").mkdir(parents=True)
            (root / "storage/part/m1").mkdir(parents=True)
            (root / "storage/part/m2").mkdir(parents=True)
            (root / "storage/session/proj/s1.json").write_text(json.dumps(
                {"id": "s1", "title": "Test", "directory": "/work/demo", "time": {"created": 1}}))
            (root / "storage/message/s1/m1.json").write_text(json.dumps({"id": "m1", "role": "user"}))
            (root / "storage/part/m1/p1.json").write_text(json.dumps({"type": "text", "text": "hello"}))
            (root / "storage/message/s1/m2.json").write_text(json.dumps({"id": "m2", "role": "assistant"}))
            (root / "storage/part/m2/p2.json").write_text(json.dumps({"type": "text", "text": "hi"}))
            sessions = list(discover_opencode(root))
            self.assertEqual(sessions[0].session.requests[0].assistant_text, "hi")

    def test_opencode_sqlite_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = sqlite3.connect(root / "opencode.db")
            db.execute("create table session (id text, directory text, title text, model text, time_created integer)")
            db.execute("create table message (id text, session_id text, time_created integer, data text)")
            db.execute("create table part (id text, message_id text, data text)")
            db.execute("insert into session values (?, ?, ?, ?, ?)",
                       ("s1", "/work/demo", "Test", '{"id":"model"}', 1))
            db.executemany("insert into message values (?, ?, ?, ?)", [
                ("m1", "s1", 2, '{"role":"user"}'), ("m2", "s1", 3, '{"role":"assistant"}')])
            db.executemany("insert into part values (?, ?, ?)", [
                ("p1", "m1", '{"type":"text","text":"hello"}'),
                ("p2", "m2", '{"type":"text","text":"hi"}')])
            db.commit()
            db.close()
            sessions = list(discover_opencode(root))
            self.assertEqual(sessions[0].session.model_id, "model")
            self.assertEqual(sessions[0].session.requests[0].assistant_text, "hi")

    def test_opencode_tool_detail_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = sqlite3.connect(root / "opencode.db")
            db.execute("create table session (id text, directory text, title text, model text, "
                       "version text, time_created integer)")
            db.execute("create table message (id text, session_id text, time_created integer, data text)")
            db.execute("create table part (id text, message_id text, data text)")
            db.execute("insert into session values (?, ?, ?, ?, ?, ?)",
                       ("s1", "/work/demo", "Test", None, "1.2.3", 1))
            assistant = {"role": "assistant", "modelID": "glm", "finish": "stop", "mode": "build",
                         "time": {"created": 100, "completed": 2600},
                         "tokens": {"input": 40, "output": 9, "reasoning": 2,
                                    "cache": {"read": 700, "write": 5}}}
            # A second call in the same turn, as every tool round makes one.
            assistant2 = {"role": "assistant", "modelID": "glm", "finish": "stop", "mode": "build",
                          "time": {"created": 2700, "completed": 3600},
                          "tokens": {"input": 5000, "output": 300, "reasoning": 3,
                                     "cache": {"read": 60000, "write": 7}}}
            db.executemany("insert into message values (?, ?, ?, ?)", [
                ("m1", "s1", 2, '{"role":"user"}'), ("m2", "s1", 3, json.dumps(assistant)),
                ("m3", "s1", 4, json.dumps(assistant2))])
            db.executemany("insert into part values (?, ?, ?)", [
                ("p1", "m1", '{"type":"text","text":"hello"}'),
                ("p2", "m2", json.dumps({"type": "tool", "tool": "bash", "callID": "c1", "state": {
                    "status": "completed", "title": "list files",
                    "input": {"command": "ls"}, "output": "a.py",
                    "metadata": {"exit": 0}, "time": {"start": 10, "end": 40}}})),
                ("p3", "m2", json.dumps({"type": "tool", "tool": "read", "callID": "c2", "state": {
                    "status": "error", "input": {"filePath": "/nope"},
                    "error": "File not found", "time": {"start": 50, "end": 60}}}))])
            db.commit()
            db.close()
            session = list(discover_opencode(root))[0].session
            req = session.requests[0]
            bash, read = req.tool_calls
            self.assertEqual((bash.command, bash.exit_code, bash.title, bash.duration_ms),
                             ("ls", 0, "list files", 30))
            self.assertEqual((read.is_error, read.output), (True, "File not found"))
            self.assertEqual((req.model_id, req.stop_reason, req.mode_name), ("glm", "stop", "build"))
            # Output, cache writes and reasoning accumulate across the calls,
            # while input and cache reads are one call's context and peak.
            self.assertEqual((req.completion_tokens, req.input_tokens), (309, 5000))
            self.assertEqual((req.cache_read_tokens, req.cache_write_tokens), (60000, 12))
            self.assertEqual(req.reasoning_tokens, 5)
            self.assertEqual(session.peak_context_tokens, 65000)
            # No model on the session row, so it falls back to the turns.
            self.assertEqual((session.model_id, session.agent_version), ("glm", "1.2.3"))


if __name__ == "__main__":
    unittest.main()
