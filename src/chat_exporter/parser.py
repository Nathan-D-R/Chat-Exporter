"""
Parse Copilot chat session JSONL files into structured Python objects.

JSONL format (event-sourced):
  kind=0  - Initial full state snapshot
  kind=1  - Single-field update  (k = key path, v = new value)
  kind=2  - Array/nested update  (k = key path, v = new value)

Key paths of interest:
  ['customTitle']                  - session title string
  ['requests']                     - single-item list with a new request
  ['requests', N, 'response']      - completed response for request N
  ['pendingRequests']              - in-flight requests (ignored for export)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ResponsePart:
    """A single rendered part of an assistant response."""
    kind: str          # 'text', 'thinking', 'tool_call', 'reference'
    text: str = ""
    tool_name: str = ""
    tool_message: str = ""


@dataclass
class ChatRequest:
    request_id: str = ""
    timestamp_ms: int = 0
    model_id: str = ""
    user_text: str = ""
    is_system_initiated: bool = False
    response_parts: list[ResponsePart] = field(default_factory=list)

    @property
    def timestamp(self) -> datetime:
        if self.timestamp_ms:
            return datetime.fromtimestamp(self.timestamp_ms / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(0, tz=timezone.utc)

    @property
    def assistant_text(self) -> str:
        """Concatenated plain text parts of the response."""
        return "\n\n".join(
            p.text for p in self.response_parts if p.kind == "text" and p.text
        )


@dataclass
class ChatSession:
    session_id: str
    title: str
    created_ms: int
    model_id: str
    requests: list[ChatRequest] = field(default_factory=list)
    provider: str = "copilot"
    workspace: str = "Unknown"
    source_path: str = ""

    @property
    def created_at(self) -> datetime:
        if self.created_ms:
            return datetime.fromtimestamp(self.created_ms / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(0, tz=timezone.utc)

    @property
    def user_requests(self) -> list[ChatRequest]:
        """Only human-initiated requests with non-empty messages."""
        return [r for r in self.requests if not r.is_system_initiated and r.user_text]


# ---------------------------------------------------------------------------
# JSONL reconstruction helpers
# ---------------------------------------------------------------------------

def _extract_text(item: Any) -> str:
    """Extract plain text from a markdown content item or string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("value") or item.get("text") or ""
    return ""


def _parse_response(raw_response: list[Any]) -> list[ResponsePart]:
    parts: list[ResponsePart] = []
    for item in raw_response:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")

        if kind is None:
            # Plain markdown text part
            text = _extract_text(item)
            if text:
                parts.append(ResponsePart(kind="text", text=text))

        elif kind == "thinking":
            text = _extract_text(item.get("value") or item.get("text") or item)
            if text:
                parts.append(ResponsePart(kind="thinking", text=text))

        elif kind == "toolInvocationSerialized":
            msg_raw = item.get("invocationMessage") or {}
            tool_msg = _extract_text(msg_raw)
            tool_id = item.get("toolId", "")
            # Also collect any tool result text
            specific = item.get("toolSpecificData") or {}
            result_text = ""
            # Terminal output
            if specific.get("kind") == "terminal":
                result_text = specific.get("output", "")
            # Generic result field
            if not result_text and isinstance(specific, dict):
                result_text = specific.get("result", specific.get("output", ""))
            if isinstance(result_text, dict):
                result_text = result_text.get("value", "")
            parts.append(ResponsePart(
                kind="tool_call",
                tool_name=tool_id,
                tool_message=tool_msg,
                text=str(result_text) if result_text else "",
            ))

        elif kind in ("inlineReference", "mcpServersStarting", "progressTaskSerialized", "questionCarousel"):
            pass  # Structural/UI items, not useful for export

    return parts


def _build_request(raw: dict[str, Any]) -> ChatRequest:
    msg = raw.get("message") or {}
    if not isinstance(msg, dict):
        msg = {}

    req = ChatRequest(
        request_id=raw.get("requestId", ""),
        timestamp_ms=raw.get("timestamp", 0),
        model_id=raw.get("modelId", ""),
        user_text=msg.get("text", ""),
        is_system_initiated=bool(raw.get("isSystemInitiated", False)),
    )
    raw_resp = raw.get("response") or []
    if isinstance(raw_resp, list):
        req.response_parts = _parse_response(raw_resp)
    return req


# ---------------------------------------------------------------------------
# Public parse function
# ---------------------------------------------------------------------------

def parse_session(path: Path) -> ChatSession | None:
    """Parse a .jsonl chat session file and return a ChatSession, or None if empty."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    # Initial state from kind=0
    state: dict[str, Any] = {}
    # Accumulated request dicts, index-addressable
    raw_requests: list[dict[str, Any]] = []

    for raw_line in lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        kind = record.get("kind")

        if kind == 0:
            state = dict(record.get("v") or {})
            # Seed requests from initial snapshot (usually empty)
            for req in state.get("requests") or []:
                if isinstance(req, dict):
                    raw_requests.append(req)

        elif kind == 1:
            keys = record.get("k") or []
            val = record.get("v")
            if len(keys) == 1:
                state[keys[0]] = val

        elif kind == 2:
            keys = record.get("k") or []
            val = record.get("v")

            if keys == ["requests"]:
                if isinstance(val, list):
                    if len(val) == 1:
                        # Single-item append pattern
                        raw_requests.append(val[0])
                    else:
                        # Full replacement (should be rare)
                        raw_requests = list(val)

            elif (
                len(keys) == 3
                and keys[0] == "requests"
                and isinstance(keys[1], int)
                and keys[2] == "response"
            ):
                idx: int = keys[1]
                # Grow array if needed (requests may arrive out-of-order)
                while len(raw_requests) <= idx:
                    raw_requests.append({})
                raw_requests[idx]["response"] = val

    if not raw_requests:
        return None

    session_id = state.get("sessionId", path.stem)
    title = state.get("customTitle", "").strip() or _derive_title(raw_requests)
    created_ms = int(state.get("creationDate") or 0)
    model_id = state.get("inputState", {}).get("selectedModel", {}).get("identifier", "")

    session = ChatSession(
        session_id=session_id,
        title=title or session_id,
        created_ms=created_ms,
        model_id=model_id,
        requests=[_build_request(r) for r in raw_requests if isinstance(r, dict)],
    )
    return session


def _derive_title(raw_requests: list[dict[str, Any]], max_len: int = 60) -> str:
    """Derive a title from the first user message when no customTitle exists."""
    for r in raw_requests:
        if r.get("isSystemInitiated"):
            continue
        text = (r.get("message") or {}).get("text", "").strip()
        if text:
            # Strip markdown and take first line
            first_line = text.splitlines()[0].strip().lstrip("#").strip()
            return first_line[:max_len] + ("..." if len(first_line) > max_len else "")
    return "Untitled Session"
