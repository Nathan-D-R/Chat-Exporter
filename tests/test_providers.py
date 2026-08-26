import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from chat_exporter.providers import discover_opencode, parse_agy, parse_claude, parse_codex


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
                {"type": "assistant", "timestamp": "2026-01-01T00:00:02Z",
                 "message": {"role": "assistant", "model": "claude-test", "stop_reason": "tool_use",
                             "usage": {"output_tokens": 10, "input_tokens": 5,
                                       "cache_read_input_tokens": 900,
                                       "cache_creation_input_tokens": 40},
                             "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                          "input": {"command": "ls", "description": "list"}},
                                         {"type": "tool_use", "id": "t2", "name": "Edit",
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
                {"type": "assistant", "timestamp": "2026-01-01T00:00:05Z",
                 "message": {"role": "assistant", "model": "claude-test", "stop_reason": "end_turn",
                             "usage": {"output_tokens": 4, "input_tokens": 6,
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
            # Output accumulates across the turn; context figures are the last call's.
            self.assertEqual((req.completion_tokens, req.input_tokens), (14, 6))
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
            db.executemany("insert into message values (?, ?, ?, ?)", [
                ("m1", "s1", 2, '{"role":"user"}'), ("m2", "s1", 3, json.dumps(assistant))])
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
            self.assertEqual((req.completion_tokens, req.input_tokens), (9, 40))
            self.assertEqual((req.cache_read_tokens, req.cache_write_tokens), (700, 5))
            self.assertEqual((req.reasoning_tokens, req.total_elapsed_ms), (2, 2500))
            # No model on the session row, so it falls back to the turns.
            self.assertEqual((session.model_id, session.agent_version), ("glm", "1.2.3"))


if __name__ == "__main__":
    unittest.main()
