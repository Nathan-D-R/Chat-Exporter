"""Flatten ChatSession objects into five tables and write them as Parquet.

Row building (`TableRows`) is plain Python and needs no extra libraries. Only
`write_parquet` imports polars, so the other output formats and the MCP server
never load it.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parser import ChatRequest, ChatSession, ToolCall, _epoch_ms
from .renderer import DEFAULT_MAX_TOOL_OUTPUT, _truncate

# Column names and types per table. Type names map to polars dtypes in
# `_polars_schema`: str, int, bool, datetime (UTC), list[str].
SESSION_COLUMNS: dict[str, str] = {
    "session_id": "str", "provider": "str", "workspace": "str", "title": "str",
    "created_utc": "datetime", "model_id": "str", "responder": "str",
    "agent_version": "str", "cwd": "str", "git_branch": "str", "source_path": "str",
    "n_turns": "int", "n_user_turns": "int", "n_tool_calls": "int",
    "n_tool_errors": "int", "models": "list[str]",
    "first_turn_utc": "datetime", "last_turn_utc": "datetime",
    "input_tokens": "int", "completion_tokens": "int",
}

TURN_COLUMNS: dict[str, str] = {
    "session_id": "str", "turn_index": "int", "request_id": "str",
    "timestamp_utc": "datetime", "is_system_initiated": "bool", "system_label": "str",
    "model_id": "str", "resolved_model": "str", "mode_name": "str",
    "permission_level": "str", "agent_version": "str", "user_text": "str",
    "assistant_text": "str", "thinking_text": "str", "warnings": "str",
    "n_tool_calls": "int", "n_tool_errors": "int", "n_subagent_calls": "int",
    "input_tokens": "int", "completion_tokens": "int", "cache_read_tokens": "int",
    "cache_write_tokens": "int", "reasoning_tokens": "int", "elapsed_ms": "int",
    "total_elapsed_ms": "int", "first_progress_ms": "int", "error_code": "str",
    "error_message": "str", "is_incomplete": "bool", "stop_reason": "str",
    "context_files": "list[str]", "edited_files": "list[str]",
}

TOOL_CALL_COLUMNS: dict[str, str] = {
    "session_id": "str", "turn_index": "int", "call_index": "int", "part_index": "int",
    "call_id": "str", "tool_id": "str", "name": "str", "title": "str",
    "message": "str", "past_tense": "str", "arguments": "str", "output": "str",
    "output_chars": "int", "is_error": "bool", "is_complete": "bool",
    "error_code": "str", "error_message": "str", "failure_category": "str",
    "failure_stage": "str", "context_tokens": "int", "context_limit": "int",
    "command": "str", "cwd": "str", "exit_code": "int", "duration_ms": "int",
    "result_files": "list[str]", "todos": "str", "is_subagent": "bool",
    "subagent_description": "str", "subagent_prompt": "str",
    "subagent_model": "str", "subagent_result": "str",
}

EVENT_COLUMNS: dict[str, str] = {
    "session_id": "str", "provider": "str", "event_index": "int", "type": "str",
    "timestamp_utc": "datetime", "event_id": "str", "parent_id": "str",
    "tool_call_id": "str", "turn_id": "str", "interaction_id": "str", "model": "str",
    "data_keys": "list[str]", "data_json": "str", "data_chars": "int",
    "scalars_json": "str", "nested_json": "str",
}

PERMISSION_COLUMNS: dict[str, str] = {
    "session_id": "str", "provider": "str", "request_id": "str", "tool_call_id": "str",
    "kind": "str", "command": "str", "path": "str", "url": "str", "tool_name": "str",
    "request_json": "str", "requested_utc": "datetime", "completed_utc": "datetime",
    "wait_ms": "int", "result_kind": "str", "decision_source": "str", "result_json": "str",
    "agent_mode": "str", "permission_mode": "str", "recovery_episode_id": "str",
    "assisted_assessment_json": "str", "approved": "bool", "outcome": "str",
}

# Providers whose sessions come from an events.jsonl log.
EVENT_PROVIDERS = ("copilot-cli", "copilot-desktop")
DATA_JSON_LIMIT = 4000
REQUEST_JSON_LIMIT = 2000
RESULT_JSON_LIMIT = 1000
ASSESSMENT_JSON_LIMIT = 1000
SCALAR_STRING_LIMIT = 300
# Keys of `data` whose nested value is kept in events.nested_json (scalar leaves only).
NESTED_KEYS = ("result", "error", "assistedApprovalAssessment")
SAVED_RESULT_KINDS = ("approved-for-location", "approved-read-only-for-session")

log = logging.getLogger("chat_exporter.tables")

TABLES: dict[str, dict[str, str]] = {
    "sessions": SESSION_COLUMNS,
    "turns": TURN_COLUMNS,
    "tool_calls": TOOL_CALL_COLUMNS,
    "events": EVENT_COLUMNS,
    "permissions": PERMISSION_COLUMNS,
}


def _utc(ms: int) -> datetime | None:
    return _epoch_ms(ms) if ms else None


def _sum_or_none(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _todos_json(tool: ToolCall) -> str:
    if not tool.todos:
        return ""
    return json.dumps(
        [{"title": t.title, "status": t.status} for t in tool.todos], ensure_ascii=False
    )


def turn_row(
    session_id: str, index: int, req: ChatRequest, include_thinking: bool = False
) -> dict[str, Any]:
    parts = req.response_parts
    tools = req.tool_calls
    return {
        "session_id": session_id,
        "turn_index": index,
        "request_id": req.request_id,
        "timestamp_utc": _utc(req.timestamp_ms),
        "is_system_initiated": req.is_system_initiated,
        "system_label": req.system_label,
        "model_id": req.model_id,
        "resolved_model": req.resolved_model,
        "mode_name": req.mode_name,
        "permission_level": req.permission_level,
        "agent_version": req.agent_version,
        "user_text": req.user_text,
        "assistant_text": req.assistant_text,
        "thinking_text": "\n\n".join(
            p.text for p in parts if p.kind == "thinking" and p.text
        ) if include_thinking else "",
        "warnings": "\n".join(p.text for p in parts if p.kind == "warning" and p.text),
        "n_tool_calls": len(tools),
        "n_tool_errors": sum(1 for t in tools if t.is_error),
        "n_subagent_calls": sum(1 for t in tools if t.subagent is not None),
        "input_tokens": req.input_tokens,
        "completion_tokens": req.completion_tokens,
        "cache_read_tokens": req.cache_read_tokens,
        "cache_write_tokens": req.cache_write_tokens,
        "reasoning_tokens": req.reasoning_tokens,
        "elapsed_ms": req.elapsed_ms,
        "total_elapsed_ms": req.total_elapsed_ms,
        "first_progress_ms": req.first_progress_ms,
        "error_code": req.error_code,
        "error_message": req.error_message,
        "is_incomplete": req.is_incomplete,
        "stop_reason": req.stop_reason,
        "context_files": list(req.context_files),
        "edited_files": req.all_edited_files,
    }


def tool_call_rows(
    session_id: str, index: int, req: ChatRequest, max_tool_output: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for part_index, part in enumerate(req.response_parts):
        tool = part.tool
        if part.kind != "tool_call" or not tool:
            continue
        sub = tool.subagent
        rows.append({
            "session_id": session_id,
            "turn_index": index,
            "call_index": len(rows),
            "part_index": part_index,
            "call_id": tool.call_id,
            "tool_id": tool.tool_id,
            "name": tool.name,
            "title": tool.title,
            "message": tool.message,
            "past_tense": tool.past_tense,
            "arguments": tool.arguments,
            "output": _truncate(tool.output, max_tool_output),
            "output_chars": len(tool.output),
            "is_error": tool.is_error,
            "is_complete": tool.is_complete,
            "error_code": tool.error_code,
            "error_message": tool.error_message,
            "failure_category": tool.failure_category,
            "failure_stage": tool.failure_stage,
            "context_tokens": tool.context_tokens,
            "context_limit": tool.context_limit,
            "command": tool.command,
            "cwd": tool.cwd,
            "exit_code": tool.exit_code,
            "duration_ms": tool.duration_ms,
            "result_files": list(tool.result_files),
            "todos": _todos_json(tool),
            "is_subagent": sub is not None,
            "subagent_description": sub.description if sub else "",
            "subagent_prompt": sub.prompt if sub else "",
            "subagent_model": sub.model_name if sub else "",
            "subagent_result": sub.result if sub else "",
        })
    return rows


def _parse_stamp(value: Any) -> datetime | None:
    """An ISO timestamp as an aware UTC datetime; None when absent or unparsable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _compact(value: Any, limit: int) -> tuple[str, int]:
    """Compact JSON of value cut to limit characters, and its length before the cut."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text[:limit], len(text)


def _str(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value)


def _first(source: dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = _str(source.get(key))
        if text:
            return text
    return ""


def _is_scalar(value: Any) -> bool:
    """A bool, a finite number or a string; nulls, lists and objects are not scalars."""
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, (bool, int, str))


def scalars_json(data: dict[str, Any]) -> str:
    """Compact JSON of the top-level entries of data that are a bool, a number or a
    string of at most SCALAR_STRING_LIMIT characters. Never cut, so it always parses."""
    kept = {
        key: value for key, value in data.items()
        if isinstance(key, str) and _is_scalar(value)
        and (not isinstance(value, str) or len(value) <= SCALAR_STRING_LIMIT)
    }
    return json.dumps(kept, ensure_ascii=False, separators=(",", ":"))


def nested_json(data: dict[str, Any]) -> str:
    """Compact JSON of the NESTED_KEYS values in data, each reduced to its scalar leaves
    one level down (long strings are cut). Always parses."""
    kept: dict[str, Any] = {}
    for key in NESTED_KEYS:
        value = data.get(key)
        if isinstance(value, dict):
            kept[key] = {
                k: (v[:SCALAR_STRING_LIMIT] if isinstance(v, str) else v)
                for k, v in value.items() if isinstance(k, str) and _is_scalar(v)
            }
        elif isinstance(value, str):
            kept[key] = value[:SCALAR_STRING_LIMIT]
    return json.dumps(kept, ensure_ascii=False, separators=(",", ":"))


def permission_outcome(result_kind: str, completed: bool) -> str:
    """One of approved, approved_saved, denied_by_user, denied_unattended, cancelled,
    pending or other, from a permission.completed result kind."""
    if not completed or not result_kind:
        return "pending"
    if result_kind == "approved":
        return "approved"
    if result_kind.startswith("approved"):
        return "approved_saved"
    if result_kind.startswith("denied-interactively"):
        return "denied_by_user"
    if "could-not-request" in result_kind:
        return "denied_unattended"
    if result_kind == "cancelled":
        return "cancelled"
    return "other"


def event_row(
    session: ChatSession, index: int, event: dict[str, Any]
) -> dict[str, Any]:
    data = event.get("data")
    data = data if isinstance(data, dict) else {}
    request = data.get("permissionRequest")
    request = request if isinstance(request, dict) else {}
    text, chars = _compact(data, DATA_JSON_LIMIT)
    return {
        "session_id": session.session_id,
        "provider": session.provider,
        "event_index": index,
        "type": _str(event.get("type")),
        "timestamp_utc": _parse_stamp(event.get("timestamp")),
        "event_id": _str(event.get("id")),
        "parent_id": _str(event.get("parentId")),
        "tool_call_id": _first(data, "toolCallId") or _first(request, "toolCallId"),
        "turn_id": _first(data, "turnId"),
        "interaction_id": _first(data, "interactionId"),
        "model": _first(data, "model"),
        "data_keys": sorted(data),
        "data_json": text,
        "data_chars": chars,
        "scalars_json": scalars_json(data),
        "nested_json": nested_json(data),
    }


def permission_rows(
    session: ChatSession, events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per permission.requested event, joined to its completion by requestId."""
    completed: dict[str, dict[str, Any]] = {}
    for event in events:
        data = event.get("data")
        if event.get("type") == "permission.completed" and isinstance(data, dict):
            request_id = _first(data, "requestId")
            if request_id:
                completed[request_id] = event
    rows: list[dict[str, Any]] = []
    for event in events:
        data = event.get("data")
        if event.get("type") != "permission.requested" or not isinstance(data, dict):
            continue
        request = data.get("permissionRequest")
        request = request if isinstance(request, dict) else {}
        request_id = _first(data, "requestId")
        done = completed.get(request_id) if request_id else None
        done_data = done.get("data") if done else None
        done_data = done_data if isinstance(done_data, dict) else {}
        result = done_data.get("result")
        result_dict = result if isinstance(result, dict) else {}
        requested = _parse_stamp(event.get("timestamp"))
        finished = _parse_stamp(done.get("timestamp")) if done else None
        wait = None
        if requested and finished:
            wait = int((finished - requested).total_seconds() * 1000)
        tool = _first(request, "toolName")
        server = _first(request, "serverName")
        name = f"{server}/{tool}" if server and tool else tool or server
        request_json, _ = _compact(request, REQUEST_JSON_LIMIT)
        result_json, _ = _compact(result, RESULT_JSON_LIMIT) if done and result is not None \
            else ("", 0)
        assessment = data.get("assistedApprovalAssessment")
        assessment_json = _compact(assessment, ASSESSMENT_JSON_LIMIT)[0] \
            if assessment is not None else ""
        result_kind = _first(result_dict, "kind") if done else ""
        outcome = permission_outcome(result_kind, done is not None)
        rows.append({
            "session_id": session.session_id,
            "provider": session.provider,
            "request_id": request_id,
            "tool_call_id": _first(request, "toolCallId") or _first(data, "toolCallId")
            or _first(done_data, "toolCallId"),
            "kind": _first(request, "kind"),
            "command": _first(request, "fullCommandText"),
            "path": _first(request, "path", "fileName", "filePath", "directory"),
            "url": _first(request, "url"),
            "tool_name": name,
            "request_json": request_json,
            "requested_utc": requested,
            "completed_utc": finished,
            "wait_ms": wait,
            "result_kind": result_kind,
            "decision_source": _first(done_data, "decisionSource"),
            "result_json": result_json,
            "agent_mode": _first(data, "agentMode"),
            "permission_mode": _first(data, "permissionMode"),
            "recovery_episode_id": _first(data, "recoveryEpisodeId")
            or _first(done_data, "recoveryEpisodeId"),
            "assisted_assessment_json": assessment_json,
            "approved": result_kind.startswith("approved"),
            "outcome": outcome,
        })
    return rows


def read_events(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """(0-based line index, event) pairs for the lines that parse as JSON
    objects, and the count of blank or unparsable lines that were skipped."""
    events: list[tuple[int, dict[str, Any]]] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            try:
                event = json.loads(line) if line else None
            except ValueError:
                event = None
            if isinstance(event, dict):
                events.append((index, event))
            else:
                skipped += 1
    return events, skipped


def session_row(session: ChatSession, turns: list[dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, None] = {}
    for turn in turns:
        name = turn["resolved_model"] or turn["model_id"]
        if name:
            models.setdefault(name, None)
    if not models and session.model_id:
        models[session.model_id] = None
    stamps = [r.timestamp_ms for r in session.requests if r.timestamp_ms]
    return {
        "session_id": session.session_id,
        "provider": session.provider,
        "workspace": session.workspace,
        "title": session.title,
        "created_utc": _utc(session.created_ms),
        "model_id": session.model_id,
        "responder": session.responder,
        "agent_version": session.agent_version,
        "cwd": session.cwd,
        "git_branch": session.git_branch,
        "source_path": session.source_path,
        "n_turns": len(turns),
        "n_user_turns": sum(1 for t in turns if not t["is_system_initiated"]),
        "n_tool_calls": sum(t["n_tool_calls"] for t in turns),
        "n_tool_errors": sum(t["n_tool_errors"] for t in turns),
        "models": list(models),
        "first_turn_utc": _utc(min(stamps)) if stamps else None,
        "last_turn_utc": _utc(max(stamps)) if stamps else None,
        "input_tokens": _sum_or_none([t["input_tokens"] for t in turns]),
        "completion_tokens": _sum_or_none([t["completion_tokens"] for t in turns]),
    }


@dataclass
class TableRows:
    """Rows accumulated across sessions; holds no session objects."""
    include_thinking: bool = False
    include_tools: bool = True
    max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT
    sessions: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    permissions: list[dict[str, Any]] = field(default_factory=list)

    def add_session(self, session: ChatSession) -> None:
        turns: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        for index, req in enumerate(session.requests):
            turns.append(turn_row(session.session_id, index, req, self.include_thinking))
            if self.include_tools:
                calls.extend(
                    tool_call_rows(session.session_id, index, req, self.max_tool_output)
                )
        self.sessions.append(session_row(session, turns))
        self.turns.extend(turns)
        self.tool_calls.extend(calls)
        self._add_events(session)

    def _add_events(self, session: ChatSession) -> None:
        """Event and permission rows from the session's events.jsonl, if it has one.

        One unreadable file or bad line never stops the export."""
        if session.provider not in EVENT_PROVIDERS or not session.source_path:
            return
        try:
            pairs, skipped = read_events(Path(session.source_path))
        except OSError as exc:
            log.warning("events: cannot read %s: %s", session.source_path, exc)
            return
        if skipped:
            log.warning("events: session %s skipped %d blank or unparsable line(s)",
                        session.session_id, skipped)
        self.events.extend(event_row(session, i, e) for i, e in pairs)
        self.permissions.extend(permission_rows(session, [e for _, e in pairs]))

    def table(self, name: str) -> list[dict[str, Any]]:
        return getattr(self, name)


def _polars_schema(columns: dict[str, str]) -> dict[str, Any]:
    import polars as pl

    dtypes = {
        "str": pl.String, "int": pl.Int64, "bool": pl.Boolean,
        "datetime": pl.Datetime("us", "UTC"), "list[str]": pl.List(pl.String),
    }
    return {name: dtypes[kind] for name, kind in columns.items()}


def write_parquet(rows: TableRows, output_dir: Path) -> dict[str, Path]:
    """Write the five tables (sessions, turns, tool_calls, events, permissions) as Parquet."""
    import polars as pl

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, columns in TABLES.items():
        data = rows.table(name)
        frame = pl.DataFrame(
            {col: [row[col] for row in data] for col in columns},
            schema=_polars_schema(columns),
        )
        path = output_dir / f"{name}.parquet"
        frame.write_parquet(path)
        written[name] = path
    return written
