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
from .parser import ChatRequest, ChatSession, ResponsePart, SubagentCall, ToolCall, parse_session

PROVIDERS = ("copilot", "agy", "claude", "codex", "opencode")


@dataclass
class DiscoveredSession:
    provider: str
    workspace: str
    path: Path
    session: ChatSession | None
    error: str = ""


def _ms(value: Any) -> int:
    """Epoch milliseconds from a number of seconds or milliseconds, or an ISO
    timestamp. Zero for anything missing, unparseable, or before the epoch -
    agy writes an `0001-01-01` sentinel where it has no time to record."""
    if isinstance(value, (int, float)):
        return max(int(value if value > 10_000_000_000 else value * 1000), 0)
    if isinstance(value, str) and value:
        try:
            return max(int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000), 0)
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


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int | None:
    """Ints only - bools are ints in Python and never a useful count here."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


# ---------------------------------------------------------------------------
# claude  (~/.claude/projects/**/*.jsonl)
# ---------------------------------------------------------------------------

# toolUseResult.type values that mean the call wrote to the named file.
_CLAUDE_EDIT_RESULTS = {"create", "update"}


def _claude_result_text(content: Any) -> str:
    """Text of a tool_result block: a bare string, or a list of content items."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _text(content)
    chunks: list[str] = []
    for item in content:
        item = _dict(item)
        if item.get("type") == "image":
            chunks.append("[image]")
        else:
            chunks.append(_text(item))
    return "\n\n".join(filter(None, chunks))


def _claude_agent_result(agent_files: dict[str, Path], agent_id: str) -> str:
    """Final assistant message of a subagent's own transcript, if it is on disk."""
    path = agent_files.get(agent_id)
    if path is None:
        return ""
    result = ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("type") != "assistant":
            continue
        text = "\n\n".join(
            _dict(part).get("text", "")
            for part in _dict(item.get("message")).get("content") or []
            if _dict(part).get("type") == "text"
        ).strip()
        if text:
            result = text
    return result


def _claude_tool_result(
    block: dict[str, Any],
    extra: dict[str, Any],
    tools: dict[str, ToolCall],
    request: ChatRequest,
    agent_files: dict[str, Path],
) -> None:
    """Fold a tool_result block, and the structured toolUseResult that rides
    along with it on the same record, back into the ToolCall it answers."""
    tool = tools.get(block.get("tool_use_id") or "")
    if tool is None:
        return
    tool.output = _claude_result_text(block.get("content"))
    tool.is_error = bool(block.get("is_error"))

    if extra.get("interrupted"):
        tool.is_complete = False
    if not tool.output:
        # Bash keeps its output here rather than in the result block.
        tool.output = "\n".join(
            part for part in (extra.get("stdout"), extra.get("stderr")) if isinstance(part, str)
        ).strip()
    tool.duration_ms = _int(extra.get("durationMs")) or tool.duration_ms

    file_path = extra.get("filePath")
    if isinstance(file_path, str) and (
        extra.get("type") in _CLAUDE_EDIT_RESULTS or extra.get("structuredPatch")
    ):
        request.edited_files.append(file_path)

    agent_id = extra.get("agentId")
    if isinstance(agent_id, str) and agent_id:
        tool.subagent = SubagentCall(
            # Synchronous launches record the description only on the call.
            description=str(extra.get("description") or "") or tool.message,
            prompt=str(extra.get("prompt") or ""),
            model_name=str(extra.get("resolvedModel") or ""),
            result=_claude_agent_result(agent_files, agent_id),
        )


def _claude_assistant(
    item: dict[str, Any],
    request: ChatRequest,
    tools: dict[str, ToolCall],
    usage_seen: dict[str, int],
) -> None:
    message = _dict(item.get("message"))
    request.model_id = message.get("model") or request.model_id
    stop = message.get("stop_reason")
    if stop:
        request.stop_reason = str(stop)

    # A turn is many API calls, and each call is written as one record per
    # content block, every record repeating that call's usage. Counting per
    # record therefore charges a response once per block: across this corpus
    # 1,641 of 2,090 message ids repeat, inflating output tokens ~2.3x.
    #
    # So accumulate per message id, not per record. A repeated id usually
    # carries an identical count, but streaming can revise it upward, and the
    # final record holds the larger value in nearly every observed case; take
    # the last value seen for an id and correct the running total by the
    # difference. Input and cache figures are the context size of a single
    # call, so the last call of the turn is the meaningful one either way.
    usage = _dict(message.get("usage"))
    output_tokens = _int(usage.get("output_tokens"))
    message_id = message.get("id")
    if output_tokens is not None:
        if isinstance(message_id, str):
            previous = usage_seen.get(message_id)
            usage_seen[message_id] = output_tokens
            delta = output_tokens - previous if previous is not None else output_tokens
        else:
            delta = output_tokens
        request.completion_tokens = (request.completion_tokens or 0) + delta
    for attr, key in (
        ("input_tokens", "input_tokens"),
        ("cache_read_tokens", "cache_read_input_tokens"),
        ("cache_write_tokens", "cache_creation_input_tokens"),
    ):
        value = _int(usage.get(key))
        if value is not None:
            setattr(request, attr, value)

    content = message.get("content", "")
    for part in content if isinstance(content, list) else [content]:
        if isinstance(part, str):
            request.response_parts.append(ResponsePart("text", text=part))
            continue
        part = _dict(part)
        kind = part.get("type")
        if kind == "text":
            request.response_parts.append(ResponsePart("text", text=part.get("text", "")))
        elif kind == "thinking":
            request.response_parts.append(ResponsePart("thinking", text=part.get("thinking", "")))
        elif kind == "tool_use":
            arguments = _dict(part.get("input"))
            tool = ToolCall(
                call_id=part.get("id", ""),
                name=part.get("name", ""),
                arguments=json.dumps(arguments, ensure_ascii=False, indent=2),
                message=str(arguments.get("description") or ""),
                command=arguments.get("command") if isinstance(arguments.get("command"), str) else "",
            )
            tools[tool.call_id] = tool
            request.response_parts.append(ResponsePart("tool_call", tool=tool))


def parse_claude(path: Path, agent_files: dict[str, Path] | None = None) -> ChatSession | None:
    """Parse one Claude Code transcript.

    `agent_files` maps an agentId to the `agent-<id>.jsonl` transcript that
    subagent wrote, so a Task/Agent call can carry its subagent's final report.
    Those files live under the project directory of the subagent's own cwd, not
    next to the transcript that launched it, so the caller builds the map.
    """
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (item.get("type") in ("user", "assistant") and not item.get("isSidechain")
                and not item.get("isMeta")):
            records.append(item)
    if agent_files is None:
        agent_files = _claude_agent_files(path.parent)

    requests: list[ChatRequest] = []
    current: ChatRequest | None = None
    tools: dict[str, ToolCall] = {}
    # message id -> last output_tokens seen, so repeated records are not double counted
    usage_seen: dict[str, int] = {}
    model = ""
    for item in records:
        message = _dict(item.get("message"))
        role = message.get("role") or item.get("type")
        content = message.get("content", "")
        if role == "user":
            results = [
                block for block in (content if isinstance(content, list) else [])
                if _dict(block).get("type") == "tool_result"
            ]
            if results:
                # Tool results arrive as user messages but belong to the turn
                # already in flight, not to a new one.
                for block in results:
                    if current is not None:
                        _claude_tool_result(
                            block, _dict(item.get("toolUseResult")), tools, current, agent_files
                        )
                continue
            text = _text(content)
            if text:
                current = ChatRequest(
                    request_id=item.get("uuid", ""), timestamp_ms=_ms(item.get("timestamp")),
                    user_text=text,
                )
                requests.append(current)
        elif role == "assistant" and current:
            model = message.get("model") or model
            _claude_assistant(item, current, tools, usage_seen)
            end_ms = _ms(item.get("timestamp"))
            if end_ms and current.timestamp_ms:
                current.total_elapsed_ms = max(end_ms - current.timestamp_ms, 0)
    if not requests:
        return None
    first = records[0]
    workspace = Path(first.get("cwd") or path.parent.name).name
    session = ChatSession(path.stem, _title(requests, path.stem), requests[0].timestamp_ms,
                          model, requests, "claude", workspace, str(path))
    session.cwd = str(first.get("cwd") or "")
    session.git_branch = str(first.get("gitBranch") or "")
    session.agent_version = str(first.get("version") or "")
    return session


def _claude_agent_files(root: Path) -> dict[str, Path]:
    return {path.stem.removeprefix("agent-"): path for path in root.rglob("agent-*.jsonl")}


def discover_claude(root: Path | None = None) -> Iterable[DiscoveredSession]:
    root = root or Path.home() / ".claude" / "projects"
    if not root.is_dir():
        return
    agent_files = _claude_agent_files(root)
    for path in sorted(root.rglob("*.jsonl")):
        try:
            session = parse_claude(path, agent_files)
            workspace = session.workspace if session else path.parent.name
            yield DiscoveredSession("claude", workspace, path, session)
        except OSError as exc:
            yield DiscoveredSession("claude", path.parent.name, path, None, str(exc))


def _codex_usage(request: ChatRequest, info: dict[str, Any]) -> None:
    """Apply one token_count event to the turn it was emitted during.

    `last_token_usage` covers the single model call that just finished, so the
    counts accumulate across the several calls that make up one turn.
    """
    usage = _dict(info.get("last_token_usage"))
    for attr, key in (
        ("completion_tokens", "output_tokens"),
        ("input_tokens", "input_tokens"),
        ("cache_read_tokens", "cached_input_tokens"),
        ("cache_write_tokens", "cache_write_input_tokens"),
        ("reasoning_tokens", "reasoning_output_tokens"),
    ):
        value = _int(usage.get(key))
        if value is not None:
            setattr(request, attr, (getattr(request, attr) or 0) + value)


def _codex_changed_paths(changes: Any) -> list[str]:
    """Paths from a FileChange item, which may key them or list them.

    The thread-item schema carries `changes` either as a path-keyed object or
    as a list of {path, kind} entries; reading only the first shape made the
    feature a silent no-op against the second.
    """
    if isinstance(changes, dict):
        return [str(name) for name in changes]
    if isinstance(changes, list):
        paths = []
        for entry in changes:
            path = _dict(entry).get("path") if isinstance(entry, dict) else entry
            if path:
                paths.append(str(path))
        return paths
    return []


def _codex_command(tool: ToolCall, item: dict[str, Any], event: dict[str, Any]) -> None:
    """Fill in a shell call from the CommandExecution event that follows it.

    The event carries its own `exec-<uuid>` id rather than the call_id, so the
    join is positional: the event is emitted immediately after the call that
    ran it, and calls that shell out to something else (a web search, say) get
    an event of a different type that matches nothing here.
    """
    command = item.get("command")
    tool.command = " ".join(command) if isinstance(command, list) else str(command or "")
    tool.cwd = str(item.get("cwd") or "")
    tool.exit_code = _int(item.get("exit_code"))
    tool.is_complete = item.get("status") != "in_progress"
    if item.get("status") == "failed":
        tool.is_error = True
    output = item.get("aggregated_output") or item.get("stdout") or ""
    if isinstance(output, str) and output.strip():
        tool.output = output
    started, completed = _int(event.get("started_at_ms")), _int(event.get("completed_at_ms"))
    if started is not None and completed is not None and completed >= started:
        tool.duration_ms = completed - started


def parse_codex(path: Path) -> ChatSession | None:
    requests: list[ChatRequest] = []
    current: ChatRequest | None = None
    meta: dict[str, Any] = {}
    model = ""
    tool_calls: dict[str, ResponsePart] = {}
    last_tool: ToolCall | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = _dict(row.get("payload"))
        row_type = row.get("type")
        if row_type == "session_meta":
            meta = payload
            continue
        if row_type == "turn_context":
            # Emitted once per turn, and the only per-turn record of the model.
            model = str(payload.get("model") or "") or model
            if current is not None and not current.model_id:
                current.model_id = model
            continue
        if row_type == "event_msg":
            kind = payload.get("type")
            if current is None:
                continue
            if kind == "token_count":
                _codex_usage(current, _dict(payload.get("info")))
            elif kind == "task_complete":
                current.total_elapsed_ms = _int(payload.get("duration_ms"))
                current.first_progress_ms = _int(payload.get("time_to_first_token_ms"))
            elif kind == "item_completed":
                item = _dict(payload.get("item"))
                if item.get("type") == "CommandExecution":
                    # The event carries its own exec-<uuid> id rather than the
                    # call_id, so the join is positional: it follows the call
                    # that ran it. Match on call_id when the event does carry
                    # one, and otherwise only consume a call still awaiting its
                    # command, so an event from a shell invocation recorded
                    # some other way cannot overwrite an unrelated call.
                    target = None
                    event_call_id = item.get("call_id") or payload.get("call_id")
                    if event_call_id:
                        part = tool_calls.get(event_call_id)
                        target = part.tool if part and part.tool else None
                    if target is None and last_tool is not None and not last_tool.command:
                        target = last_tool
                    if target is not None:
                        _codex_command(target, item, payload)
                    last_tool = None
                elif item.get("type") == "FileChange":
                    current.edited_files.extend(_codex_changed_paths(item.get("changes")))
            continue
        if row_type != "response_item":
            continue
        kind = payload.get("type")
        if kind == "message":
            role = payload.get("role")
            text = _text(payload.get("content"))
            if role == "user" and text:
                current = ChatRequest(timestamp_ms=_ms(row.get("timestamp")), user_text=text,
                                      model_id=model)
                requests.append(current)
                last_tool = None
            elif role == "assistant" and current and text:
                current.response_parts.append(ResponsePart("text", text=text))
        elif kind in ("function_call", "custom_tool_call") and current:
            tool = ToolCall(
                call_id=payload.get("call_id", ""),
                name=payload.get("name", ""),
                arguments=str(payload.get("arguments") or payload.get("input") or ""),
            )
            part = ResponsePart("tool_call", tool=tool)
            current.response_parts.append(part)
            tool_calls[tool.call_id] = part
            last_tool = tool
        elif kind in ("function_call_output", "custom_tool_call_output"):
            part = tool_calls.get(payload.get("call_id", ""))
            if part and part.tool and not part.tool.output:
                part.tool.output = _text(payload.get("output"))
        elif kind == "reasoning" and current:
            thinking = _text(payload.get("summary"))
            if thinking:
                current.response_parts.append(ResponsePart("thinking", text=thinking))
    if not requests:
        return None
    sid = meta.get("id") or meta.get("session_id") or path.stem.rsplit("-", 1)[-1]
    workspace = Path(meta.get("cwd") or "Unknown").name
    session = ChatSession(sid, _title(requests, sid), _ms(meta.get("timestamp")), model,
                          requests, "codex", workspace, str(path))
    session.cwd = str(meta.get("cwd") or "")
    session.git_branch = str(_dict(meta.get("git")).get("branch") or "")
    session.agent_version = str(meta.get("cli_version") or "")
    return session


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


def _opencode_tool(part: dict[str, Any]) -> ToolCall:
    state = _dict(part.get("state"))
    arguments = _dict(state.get("input"))
    metadata = _dict(state.get("metadata"))
    timing = _dict(state.get("time"))
    tool = ToolCall(
        call_id=str(part.get("callID") or ""),
        name=str(part.get("tool") or ""),
        title=str(state.get("title") or ""),
        arguments=json.dumps(arguments, ensure_ascii=False, indent=2),
        output=_text(state.get("output")),
        command=arguments.get("command") if isinstance(arguments.get("command"), str) else "",
        exit_code=_int(metadata.get("exit")),
        is_complete=state.get("status") != "running",
    )
    if state.get("status") == "error":
        tool.is_error = True
        tool.output = tool.output or _text(state.get("error")) or str(state.get("error") or "")
    start, end = _int(timing.get("start")), _int(timing.get("end"))
    if start is not None and end is not None and end >= start:
        tool.duration_ms = end - start
    return tool


def _opencode_usage(request: ChatRequest, info: dict[str, Any]) -> None:
    cache = _dict(info.get("cache"))
    for attr, source, key in (
        ("completion_tokens", info, "output"),
        ("input_tokens", info, "input"),
        ("reasoning_tokens", info, "reasoning"),
        ("cache_read_tokens", cache, "read"),
        ("cache_write_tokens", cache, "write"),
    ):
        value = _int(source.get(key))
        if value is not None:
            setattr(request, attr, (getattr(request, attr) or 0) + value)


def _opencode_message(info: dict[str, Any], parts: list[dict[str, Any]], current: ChatRequest | None) -> ChatRequest | None:
    role = info.get("role")
    timing = _dict(info.get("time"))
    text_parts = [p.get("text", "") for p in parts if p.get("type") == "text" and p.get("text")]
    if role == "user":
        return ChatRequest(request_id=info.get("id", ""), timestamp_ms=_ms(timing.get("created")),
                           model_id=(info.get("model") or {}).get("modelID", ""), user_text="\n\n".join(text_parts))
    if role == "assistant" and current:
        current.model_id = str(info.get("modelID") or "") or current.model_id
        current.stop_reason = str(info.get("finish") or "") or current.stop_reason
        current.mode_name = str(info.get("mode") or info.get("agent") or "") or current.mode_name
        _opencode_usage(current, _dict(info.get("tokens")))
        created, completed = _int(timing.get("created")), _int(timing.get("completed"))
        if created is not None and completed is not None and completed >= created:
            current.total_elapsed_ms = (current.total_elapsed_ms or 0) + (completed - created)
        for part in parts:
            kind = part.get("type")
            if kind == "text" and part.get("text"):
                current.response_parts.append(ResponsePart("text", text=part["text"]))
            elif kind == "reasoning" and part.get("text"):
                current.response_parts.append(ResponsePart("thinking", text=part["text"]))
            elif kind == "tool":
                tool = _opencode_tool(part)
                current.response_parts.append(ResponsePart("tool_call", tool=tool))
                path = _dict(_dict(part.get("state")).get("input")).get("filePath")
                if isinstance(path, str) and tool.name in ("write", "edit", "patch"):
                    current.edited_files.append(path)
    return current


def _opencode_session_meta(session: ChatSession, info: dict[str, Any], directory: str) -> None:
    session.cwd = str(directory or "")
    session.agent_version = str(info.get("version") or "")
    if not session.model_id:
        session.model_id = next((r.model_id for r in reversed(session.requests) if r.model_id), "")


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
            _opencode_session_meta(session, info, directory)
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
                        # The row's own timestamps are the fallback; the blob
                        # carries `time.completed`, which a blanket overwrite
                        # would drop along with per-turn duration.
                        timing = _dict(message.get("time"))
                        timing.setdefault("created", message_row["time_created"])
                        message.update(id=message_row["id"], time=timing)
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
                    _opencode_session_meta(session, info, info.get("directory") or "")
                    yield DiscoveredSession("opencode", workspace, path, session)
        except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            yield DiscoveredSession("opencode", "Unknown", path, None, str(exc))


def discover(providers: set[str], config_root: Path | None = None) -> list[DiscoveredSession]:
    adapters = {
        "copilot": lambda: discover_copilot(config_root), "agy": discover_agy,
        "claude": discover_claude, "codex": discover_codex, "opencode": discover_opencode,
    }
    return [item for name in PROVIDERS if name in providers for item in adapters[name]()]
