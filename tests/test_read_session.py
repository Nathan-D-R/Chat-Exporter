import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from chat_exporter import search
from chat_exporter.parser import ChatRequest, ChatSession, ResponsePart, ToolCall
from chat_exporter.providers import DiscoveredSession, parse_claude, parse_copilot_cli
from chat_exporter.renderer import render_session

REPEATED = "The reconciliation covers every module in the project and found no gaps."


def _claude_session(tmp: str) -> ChatSession:
    rows = [
        {"type": "user", "uuid": "u1", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/work/demo",
         "message": {"role": "user", "content": "start the review"}},
        {"type": "assistant", "timestamp": "2026-01-01T00:00:05Z", "message": {
            "role": "assistant", "model": "m", "content": [
                {"type": "text", "text": "Looking at the files."},
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "ls -la src", "description": "list"}},
                {"type": "tool_use", "id": "t2", "name": "Agent",
                 "input": {"description": "scan the logs", "prompt": "go"}}]}},
        {"type": "user", "timestamp": "2026-01-01T00:00:06Z", "message": {
            "role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "TOOL-OUTPUT-BODY"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "Found two errors " + "x" * 400}]}},
        {"type": "assistant", "timestamp": "2026-01-01T00:00:08Z", "message": {
            "role": "assistant", "model": "m", "content": [{"type": "text", "text": REPEATED}]}},
        {"type": "user", "uuid": "u2", "timestamp": "2026-01-01T00:01:00Z", "message": {
            "role": "user", "content": "<task-notification>\n<task-id>b1</task-id>\n"
            "<status>completed</status>\n<summary>Background command \"wait\" finished</summary>\n"
            "</task-notification>"}},
        {"type": "assistant", "timestamp": "2026-01-01T00:01:02Z", "message": {
            "role": "assistant", "model": "m", "content": [{"type": "text", "text": "Background work is done."}]}},
        {"type": "user", "uuid": "u3", "timestamp": "2026-01-01T00:02:00Z",
         "message": {"role": "user", "content": "next question"}},
        {"type": "assistant", "timestamp": "2026-01-01T00:02:02Z", "message": {
            "role": "assistant", "model": "m", "content": [{"type": "text", "text": REPEATED}]}},
    ]
    path = Path(tmp) / "claude-1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    session = parse_claude(path, {})
    assert session is not None
    return session


def _copilot_cli_session(tmp: str) -> ChatSession:
    events = [
        {"type": "session.start", "timestamp": "2026-08-27T01:44:55Z", "data": {
            "sessionId": "cli-1", "startTime": "2026-08-27T01:44:55Z",
            "context": {"cwd": "/work/demo"}}},
        {"type": "user.message", "timestamp": "2026-08-27T01:45:00Z",
         "data": {"content": "list it", "interactionId": "i1"}},
        {"type": "tool.execution_start", "data": {
            "toolCallId": "c1", "toolName": "bash", "arguments": {"command": "ls -a"}}},
        {"type": "tool.execution_complete", "data": {
            "toolCallId": "c1", "success": False, "result": {"content": "TOOL-OUTPUT-BODY"}}},
        {"type": "tool.execution_start", "data": {
            "toolCallId": "c2", "toolName": "task",
            "arguments": {"description": "explore the repo", "prompt": "p"}}},
        {"type": "tool.execution_complete", "data": {
            "toolCallId": "c2", "success": True, "result": {"content": "Repo has three folders"}}},
        {"type": "assistant.message", "timestamp": "2026-08-27T01:45:02Z",
         "data": {"model": "m", "content": "Two files."}},
    ]
    path = Path(tmp) / "cli-1" / "events.jsonl"
    path.parent.mkdir()
    path.write_text("\n".join(json.dumps(e) for e in events))
    session = parse_copilot_cli(path, {})
    assert session is not None
    return session


def _synthetic(turns: int, size: int = 300) -> ChatSession:
    requests = [
        ChatRequest(
            request_id=f"r{i}", timestamp_ms=1_775_178_266_000 + i * 1000,
            user_text=f"question number {i}",
            response_parts=[ResponsePart("text", text=f"answer {i} " + "word " * size)],
        )
        for i in range(1, turns + 1)
    ]
    return ChatSession("synth", "Synthetic", 1_775_178_266_000, "m", requests, "claude", "demo")


def _read(session: ChatSession, **kwargs):
    item = DiscoveredSession(session.provider, session.workspace, Path("synth.jsonl"), session)
    with mock.patch.object(search, "iter_discover", return_value=[item]):
        return search.read_session(session.provider, session.session_id, **kwargs)


class DetailLevelTests(unittest.TestCase):
    def test_claude_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = _claude_session(tmp)
        condensed = render_session(session, detail="condensed")
        self.assertIn("start the review", condensed)
        self.assertIn("Looking at the files.", condensed)
        self.assertIn("[subagent ok] scan the logs -- Found two errors", condensed)
        self.assertIn("[background task completed] Background command", condensed)
        self.assertNotIn("tool Bash", condensed)
        self.assertNotIn("TOOL-OUTPUT-BODY", condensed)
        self.assertNotIn("Tokens", condensed)
        self.assertEqual(condensed.count("### Turn"), 2)  # the notification joins turn 1
        self.assertLess(len(condensed.split("[subagent ok]")[1].split("\n")[0]), 260)

        standard = render_session(session, detail="standard")
        self.assertIn("- tool Bash: ls -la src [ok]", standard)
        self.assertIn("[subagent ok]", standard)
        self.assertNotIn("TOOL-OUTPUT-BODY", standard)

        full = render_session(session, detail="full")
        self.assertIn("TOOL-OUTPUT-BODY", full)
        self.assertEqual(full, render_session(session))  # exporter default is unchanged

    def test_copilot_cli_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = _copilot_cli_session(tmp)
        condensed = render_session(session, detail="condensed")
        self.assertIn("[subagent ok] explore the repo -- Repo has three folders", condensed)
        self.assertNotIn("bash", condensed)
        standard = render_session(session, detail="standard")
        self.assertIn("- tool bash: ls -a [error]", standard)
        self.assertNotIn("TOOL-OUTPUT-BODY", standard)
        self.assertIn("TOOL-OUTPUT-BODY", render_session(session, detail="full"))

    def test_duplicate_paragraphs_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = _claude_session(tmp)
        for level in ("condensed", "standard"):
            text = render_session(session, detail=level)
            self.assertEqual(text.count(REPEATED), 1)
            self.assertEqual(text.count("[repeated from turn 1]"), 1)
        self.assertEqual(render_session(session, detail="full").count(REPEATED), 2)

    def test_elided_paragraph_is_not_a_repeat_target(self):
        marked = "Unique marker paragraph that sits in the elided middle of a long turn."
        filler = "\n\n".join(f"Filler paragraph number {i} with enough words to count." for i in range(60))
        session = _synthetic(0)
        texts = [f"{filler}\n\n{marked}\n\n{filler}", marked]
        for i, text in enumerate(texts):
            session.requests.append(ChatRequest(
                timestamp_ms=1_775_178_266_000 + i, user_text=f"q{i}",
                response_parts=[ResponsePart("text", text=text)]))
        out = render_session(session, detail="condensed", max_assistant_chars=400)
        self.assertNotIn("[repeated from turn", out)
        self.assertIn(marked, out.split("### Turn 2")[1])
        # when the first copy is shown whole, the pointer is used
        session.requests[0].response_parts[0].text = marked
        out = render_session(session, detail="condensed", max_assistant_chars=400)
        self.assertIn("[repeated from turn 1]", out)

    def test_turn_numbers_match_across_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = _claude_session(tmp)
        # a notification before any user prompt
        lead = ChatRequest(timestamp_ms=1_700_000_000_000, is_notification=True, user_text=(
            "<task-notification><status>failed</status><summary>Early job</summary>"
            "</task-notification>"))
        session.requests.insert(0, lead)
        numbers = []
        for level in ("condensed", "standard", "full"):
            out = _read(session, detail=level)["content"]
            numbers.append(re.findall(r"### Turn (\d+)", out))
            self.assertNotIn("<task-notification>", out)
            self.assertIn("[background task failed] Early job", out)
            self.assertNotIn("**User**\n\n- [background", out)
        self.assertEqual(numbers[0], numbers[1])
        self.assertEqual(numbers[0], numbers[2])
        self.assertEqual(numbers[0], ["1", "2", "3"])

    def test_export_path_numbers_notifications_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = _claude_session(tmp)
        out = render_session(session)
        self.assertEqual(out.count("### Turn"), 3)
        self.assertIn("<task-notification>", out)

    def test_missing_timestamps_give_unknown(self):
        session = _synthetic(1, size=5)
        session.requests[0].timestamp_ms = 0
        self.assertIsNone(session.last_activity_at)
        self.assertIn("**Last activity:** unknown", render_session(session, detail="condensed"))
        item = DiscoveredSession("claude", "demo", Path("s.jsonl"), session)
        self.assertIsNone(search.search_discovered([item], "")["results"][0]["last_activity_at"])

    def test_short_paragraphs_are_not_replaced(self):
        session = _synthetic(0)
        for i in range(2):
            session.requests.append(ChatRequest(
                timestamp_ms=1_775_178_266_000 + i, user_text=f"q{i}",
                response_parts=[ResponsePart("text", text="Done, thanks.")]))
        self.assertEqual(render_session(session, detail="condensed").count("Done, thanks."), 2)

    def test_legacy_condensed_mapping(self):
        session = _synthetic(2, size=5)
        self.assertEqual(_read(session)["detail"], "condensed")
        self.assertEqual(_read(session, condensed=True)["detail"], "condensed")
        self.assertEqual(_read(session, condensed=False)["detail"], "full")
        self.assertEqual(_read(session, condensed=False, detail="standard")["detail"], "standard")
        with self.assertRaises(ValueError):
            _read(session, detail="verbose")


class PagingTests(unittest.TestCase):
    def test_from_end_keeps_last_whole_turns(self):
        session = _synthetic(10)
        result = _read(session, max_chars=3_000)
        content = result["content"]
        self.assertTrue(result["truncated"])
        self.assertEqual(result["page"], 0)
        self.assertGreater(result["pages"], 1)
        self.assertLessEqual(len(content), 3_000)
        self.assertIn("# Synthetic", content)
        self.assertIn("### Turn 10", content)
        self.assertNotIn("### Turn 1\n", content)
        self.assertRegex(content, r"\[\d+ earlier turns omitted; page=1 for the previous part\]")
        # whole turns: every shown turn keeps its full answer
        for n in range(1, 11):
            if f"### Turn {n}\n" in content:
                self.assertIn((f"answer {n} " + "word " * 300).rstrip(), content)

    def test_paging_backward_covers_everything_without_overlap(self):
        session = _synthetic(12)
        first = _read(session, max_chars=3_000)
        pages = first["pages"]
        seen: list[int] = []
        for p in range(pages):
            result = _read(session, max_chars=3_000, page=p)
            self.assertEqual((result["page"], result["pages"]), (p, pages))
            self.assertLessEqual(len(result["content"]), 3_000)
            seen = [int(n) for n in re.findall(r"### Turn (\d+)", result["content"])] + seen
        self.assertEqual(seen, list(range(1, 13)))
        with self.assertRaises(ValueError):
            _read(session, max_chars=3_000, page=pages)

    def test_from_start_pages_forward(self):
        session = _synthetic(12)
        pages = _read(session, max_chars=3_000, from_end=False)["pages"]
        seen: list[int] = []
        for p in range(pages):
            result = _read(session, max_chars=3_000, from_end=False, page=p)
            if p == 0:
                self.assertIn("### Turn 1\n", result["content"])
                self.assertIn("later turns omitted; page=1 for the next part", result["content"])
            seen += [int(n) for n in re.findall(r"### Turn (\d+)", result["content"])]
        self.assertEqual(seen, list(range(1, 13)))

    def test_oversized_turn_is_reachable_and_keeps_its_end(self):
        session = _synthetic(0)
        lines = "\n".join(f"line {i:03d} of the long answer" for i in range(200))
        session.requests.append(ChatRequest(
            timestamp_ms=1_775_178_266_000, user_text="one big question",
            response_parts=[ResponsePart("text", text=lines.replace("\n", "\n\n"))]))
        first = _read(session, max_chars=2_000, max_assistant_chars=12_000)
        self.assertGreater(first["pages"], 1)
        self.assertIn("line 199", first["content"])
        self.assertIn("[start of turn truncated", first["content"])
        found = set()
        for p in range(first["pages"]):
            out = _read(session, max_chars=2_000, max_assistant_chars=12_000, page=p)["content"]
            self.assertLessEqual(len(out), 2_000)
            found |= set(re.findall(r"line (\d{3}) of", out))
        self.assertEqual(found, {f"{i:03d}" for i in range(200)})

    def test_short_session_is_one_page(self):
        result = _read(_synthetic(2, size=5))
        self.assertEqual((result["truncated"], result["page"], result["pages"]), (False, 0, 1))
        self.assertNotIn("omitted", result["content"])

    def test_search_hits_carry_activity_and_turns(self):
        session = _synthetic(3, size=5)
        item = DiscoveredSession("claude", "demo", Path("s.jsonl"), session)
        hit = search.search_discovered([item], "question")["results"][0]
        self.assertEqual(hit["turns"], 3)
        self.assertTrue(hit["last_activity_at"].startswith("2026-"))


if __name__ == "__main__":
    unittest.main()
