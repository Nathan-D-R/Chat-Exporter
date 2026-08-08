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
                 "cwd": "/work/demo", "message": {"role": "user", "content": "hello"}},
                {"type": "assistant", "message": {"role": "assistant", "model": "claude-test",
                 "content": [{"type": "text", "text": "hi"},
                             {"type": "tool_use", "name": "Read", "input": {"path": "x"}}]}},
            ]
            path.write_text("\n".join(json.dumps(x) for x in rows))
            session = parse_claude(path)
            self.assertEqual((session.workspace, session.model_id), ("demo", "claude-test"))
            self.assertEqual(session.requests[0].assistant_text, "hi")

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


if __name__ == "__main__":
    unittest.main()
