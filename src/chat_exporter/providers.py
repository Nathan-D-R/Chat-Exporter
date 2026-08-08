"""Discovery and parsing adapters for supported chat providers."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .discovery import find_workspace_storage_dirs, get_workspace_name, list_chat_session_files
from .parser import ChatRequest, ChatSession, ResponsePart, parse_session

PROVIDERS = ("copilot", "agy", "claude", "codex", "opencode")


@dataclass
class DiscoveredSession:
    provider: str
    workspace: str
    path: Path
    session: ChatSession | None
    error: str = ""


def _ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value if value > 10_000_000_000 else value * 1000)
    if isinstance(value, str) and value:
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            pass
    return 0


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(filter(None, (_text(x) for x in content)))
    if isinstance(content, dict):
        if content.get("type") in ("text", "input_text", "output_text"):
            return str(content.get("text") or content.get("value") or "")
        return str(content.get("text") or content.get("value") or "")
    return ""


def _title(requests: list[ChatRequest], fallback: str) -> str:
    for request in requests:
        if request.user_text.strip():
            line = request.user_text.strip().splitlines()[0].lstrip("#").strip()
            return line[:60] + ("..." if len(line) > 60 else "")
    return fallback


def discover_copilot(config_root: Path | None = None) -> Iterable[DiscoveredSession]:
    for ws_dir in find_workspace_storage_dirs(config_root):
        workspace = get_workspace_name(ws_dir)
        for path in list_chat_session_files(ws_dir):
            session = parse_session(path)
            if session:
                session.workspace = workspace
                session.source_path = str(path)
            yield DiscoveredSession("copilot", workspace, path, session)


def parse_claude(path: Path) -> ChatSession | None:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (item.get("type") in ("user", "assistant") and not item.get("isSidechain")
                and not item.get("isMeta")):
            records.append(item)
    requests: list[ChatRequest] = []
    current: ChatRequest | None = None
    model = ""
    for item in records:
        message = item.get("message") or {}
        role = message.get("role") or item.get("type")
        content = message.get("content", "")
        if role == "user":
            text = _text(content)
            if text:
                current = ChatRequest(
                    request_id=item.get("uuid", ""), timestamp_ms=_ms(item.get("timestamp")),
                    user_text=text,
                )
                requests.append(current)
        elif role == "assistant" and current:
            model = message.get("model") or model
            current.model_id = message.get("model", "")
            for part in content if isinstance(content, list) else [content]:
                if isinstance(part, str):
                    current.response_parts.append(ResponsePart("text", text=part))
                elif isinstance(part, dict):
                    kind = part.get("type")
                    if kind == "text":
                        current.response_parts.append(ResponsePart("text", text=part.get("text", "")))
                    elif kind == "thinking":
                        current.response_parts.append(ResponsePart("thinking", text=part.get("thinking", "")))
                    elif kind == "tool_use":
                        current.response_parts.append(ResponsePart(
                            "tool_call", tool_name=part.get("name", ""),
                            tool_message=json.dumps(part.get("input", {}), ensure_ascii=False, indent=2),
                        ))
    if not requests:
        return None
    first = records[0]
    workspace = Path(first.get("cwd") or path.parent.name).name
    return ChatSession(path.stem, _title(requests, path.stem), requests[0].timestamp_ms,
                       model, requests, "claude", workspace, str(path))


def discover_claude(root: Path | None = None) -> Iterable[DiscoveredSession]:
    root = root or Path.home() / ".claude" / "projects"
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.jsonl")):
        try:
            session = parse_claude(path)
            workspace = session.workspace if session else path.parent.name
            yield DiscoveredSession("claude", workspace, path, session)
        except OSError as exc:
            yield DiscoveredSession("claude", path.parent.name, path, None, str(exc))


def parse_codex(path: Path) -> ChatSession | None:
    requests: list[ChatRequest] = []
    current: ChatRequest | None = None
    meta: dict[str, Any] = {}
    tool_calls: dict[str, ResponsePart] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = row.get("payload") or {}
        if row.get("type") == "session_meta":
            meta = payload
        if row.get("type") != "response_item":
            continue
        kind = payload.get("type")
        if kind == "message":
            role = payload.get("role")
            text = _text(payload.get("content"))
            if role == "user" and text:
                current = ChatRequest(timestamp_ms=_ms(row.get("timestamp")), user_text=text)
                requests.append(current)
            elif role == "assistant" and current and text:
                current.response_parts.append(ResponsePart("text", text=text))
        elif kind in ("function_call", "custom_tool_call") and current:
            part = ResponsePart("tool_call", tool_name=payload.get("name", ""),
                                tool_message=str(payload.get("arguments") or payload.get("input") or ""))
            current.response_parts.append(part)
            tool_calls[payload.get("call_id", "")] = part
        elif kind in ("function_call_output", "custom_tool_call_output"):
            part = tool_calls.get(payload.get("call_id", ""))
            if part:
                part.text = _text(payload.get("output"))
        elif kind == "reasoning" and current:
            thinking = _text(payload.get("summary"))
            if thinking:
                current.response_parts.append(ResponsePart("thinking", text=thinking))
    if not requests:
        return None
    sid = meta.get("id") or meta.get("session_id") or path.stem.rsplit("-", 1)[-1]
    workspace = Path(meta.get("cwd") or "Unknown").name
    return ChatSession(sid, _title(requests, sid), _ms(meta.get("timestamp")), "",
                       requests, "codex", workspace, str(path))


def discover_codex(root: Path | None = None) -> Iterable[DiscoveredSession]:
    root = root or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.jsonl")):
        session = parse_codex(path)
        yield DiscoveredSession("codex", session.workspace if session else "Unknown", path, session)


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7f) << shift
        if byte < 0x80:
            return value, pos
        shift += 7
    raise ValueError("truncated varint")


def _protobuf_strings(data: bytes, prefix: str = "", depth: int = 0) -> list[tuple[str, str]]:
    """Extract UTF-8 leaf strings from an unknown protobuf without credentials or descriptors."""
    found: list[tuple[str, str]] = []
    pos = 0
    try:
        while pos < len(data):
            tag, pos = _varint(data, pos)
            field, wire = tag >> 3, tag & 7
            path = f"{prefix}.{field}" if prefix else str(field)
            if not field or wire in (3, 4):
                return []
            if wire == 0:
                _, pos = _varint(data, pos)
            elif wire == 1:
                pos += 8
            elif wire == 5:
                pos += 4
            elif wire == 2:
                size, pos = _varint(data, pos)
                value = data[pos:pos + size]
                pos += size
                nested = _protobuf_strings(value, path, depth + 1) if depth < 7 else []
                if nested:
                    found.extend(nested)
                else:
                    try:
                        text = value.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    if text and sum(c.isprintable() or c in "\n\r\t" for c in text) / len(text) > .97:
                        found.append((path, text))
            else:
                return []
    except (ValueError, IndexError):
        return []
    return found


def parse_agy(path: Path, summary: dict[str, Any]) -> ChatSession | None:
    requests: list[ChatRequest] = []
    current: ChatRequest | None = None
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT idx, step_type, step_payload FROM steps ORDER BY idx")
        for idx, step_type, payload in rows:
            strings = dict(_protobuf_strings(payload or b""))
            if step_type == 14:
                text = strings.get("19.2") or strings.get("19.3.1") or ""
                if text:
                    current = ChatRequest(request_id=str(idx), user_text=text)
                    requests.append(current)
            elif current and step_type == 15:
                thinking = strings.get("20.3", "")
                if thinking:
                    current.response_parts.append(ResponsePart("thinking", text=thinking))
            elif current and step_type == 101:
                text = strings.get("114.1") or strings.get("114.2.2") or ""
                if text:
                    current.response_parts.append(ResponsePart("text", text=text))
    if not requests:
        return None
    workspace_uris = summary.get("workspace_uris", "")
    try:
        uris = json.loads(workspace_uris)
        workspace_uri = uris[0] if isinstance(uris, list) and uris else workspace_uris
    except json.JSONDecodeError:
        workspace_uri = workspace_uris
    workspace = Path(str(workspace_uri).removeprefix("file://")).name or "Unknown"
    created = _ms(summary.get("last_user_input_time") or summary.get("last_modified_time"))
    title = summary.get("title") or _title(requests, path.stem)
    return ChatSession(path.stem, title, created, summary.get("agent_name", ""), requests,
                       "agy", workspace, str(path))


def discover_agy(root: Path | None = None) -> Iterable[DiscoveredSession]:
    root = root or Path.home() / ".gemini" / "antigravity-cli"
    summaries = root / "conversation_summaries.db"
    conversations = root / "conversations"
    if not summaries.is_file() or not conversations.is_dir():
        return
    with sqlite3.connect(f"file:{summaries}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        summary_by_id = {r["conversation_id"]: dict(r) for r in db.execute("SELECT * FROM conversation_summaries")}
    for path in sorted(conversations.glob("*.db")):
        summary = summary_by_id.get(path.stem, {})
        session = parse_agy(path, summary)
        yield DiscoveredSession("agy", session.workspace if session else "Unknown", path, session)


def _opencode_message(info: dict[str, Any], parts: list[dict[str, Any]], current: ChatRequest | None) -> ChatRequest | None:
    role = info.get("role")
    text_parts = [p.get("text", "") for p in parts if p.get("type") == "text" and p.get("text")]
    if role == "user":
        return ChatRequest(request_id=info.get("id", ""), timestamp_ms=_ms((info.get("time") or {}).get("created")),
                           model_id=(info.get("model") or {}).get("modelID", ""), user_text="\n\n".join(text_parts))
    if role == "assistant" and current:
        for part in parts:
            kind = part.get("type")
            if kind == "text" and part.get("text"):
                current.response_parts.append(ResponsePart("text", text=part["text"]))
            elif kind == "reasoning" and part.get("text"):
                current.response_parts.append(ResponsePart("thinking", text=part["text"]))
            elif kind == "tool":
                state = part.get("state") or {}
                current.response_parts.append(ResponsePart("tool_call", tool_name=part.get("tool", ""),
                    tool_message=json.dumps(state.get("input", {}), ensure_ascii=False, indent=2),
                    text=_text(state.get("output"))))
    return current


def discover_opencode(root: Path | None = None) -> Iterable[DiscoveredSession]:
    root = root or Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "opencode"
    storage = root / "storage"
    seen: set[str] = set()
    for path in sorted((storage / "session").glob("*/*.json")) if storage.is_dir() else []:
        try:
            info = json.loads(path.read_text())
            sid = info.get("id") or path.stem
            requests: list[ChatRequest] = []
            message_dir = storage / "message" / sid
            for message_path in sorted(message_dir.glob("*.json")):
                message = json.loads(message_path.read_text())
                parts = [json.loads(p.read_text()) for p in sorted((storage / "part" / message_path.stem).glob("*.json"))]
                current = requests[-1] if requests else None
                result = _opencode_message(message, parts, current)
                if result is not current and result:
                    requests.append(result)
            if not requests:
                continue
            directory = info.get("directory") or info.get("path") or path.parent.name
            session = ChatSession(sid, info.get("title") or _title(requests, sid),
                _ms((info.get("time") or {}).get("created")), "", requests,
                "opencode", Path(directory).name, str(path))
            yield DiscoveredSession("opencode", session.workspace, path, session)
            seen.add(sid)
        except (OSError, json.JSONDecodeError) as exc:
            yield DiscoveredSession("opencode", path.parent.name, path, None, str(exc))
    for path in sorted(root.glob("opencode*.db")):
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
                db.row_factory = sqlite3.Row
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {"session", "message", "part"}.issubset(tables):
                    continue
                for row in db.execute("SELECT * FROM session ORDER BY time_created"):
                    info = dict(row)
                    sid = info["id"]
                    if sid in seen:
                        continue
                    requests: list[ChatRequest] = []
                    messages = db.execute(
                        "SELECT * FROM message WHERE session_id=? ORDER BY time_created, id", (sid,)
                    ).fetchall()
                    for message_row in messages:
                        message = json.loads(message_row["data"])
                        message.update(id=message_row["id"], time={"created": message_row["time_created"]})
                        part_rows = db.execute(
                            "SELECT data FROM part WHERE message_id=? ORDER BY id", (message_row["id"],)
                        ).fetchall()
                        parts = [json.loads(part_row["data"]) for part_row in part_rows]
                        current = requests[-1] if requests else None
                        result = _opencode_message(message, parts, current)
                        if result is not current and result:
                            requests.append(result)
                    if not requests:
                        continue
                    workspace = Path(info.get("directory") or "Unknown").name
                    model_data = json.loads(info["model"]) if info.get("model") else {}
                    session = ChatSession(sid, info.get("title") or _title(requests, sid),
                        _ms(info.get("time_created")), model_data.get("id", ""), requests,
                        "opencode", workspace, str(path))
                    yield DiscoveredSession("opencode", workspace, path, session)
        except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            yield DiscoveredSession("opencode", "Unknown", path, None, str(exc))


def discover(providers: set[str], config_root: Path | None = None) -> list[DiscoveredSession]:
    adapters = {
        "copilot": lambda: discover_copilot(config_root), "agy": discover_agy,
        "claude": discover_claude, "codex": discover_codex, "opencode": discover_opencode,
    }
    return [item for name in PROVIDERS if name in providers for item in adapters[name]()]
