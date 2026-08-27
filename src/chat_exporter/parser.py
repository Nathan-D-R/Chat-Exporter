"""
Parse Copilot chat session JSONL files into structured Python objects.

JSONL format (event-sourced):
  kind=0  - Initial full state snapshot (v = whole session object)
  kind=1  - Scalar/field update         (k = key path, v = new value)
  kind=2  - Array/nested update         (k = key path, v = new value)

Key paths seen in real sessions:
  ['customTitle']                       - session title string
  ['requests']                          - 1-item list appends a request,
                                          longer list replaces the array
  ['requests', N, 'response']           - response parts for request N
  ['requests', N, 'result']             - timings, token usage, error details
  ['requests', N, 'completionTokens']   - completion token count
  ['requests', N, 'elapsedMs']          - wall-clock duration
  ['requests', N, 'contentReferences']  - files pulled into context
  ['inputState', ...]                   - editor UI state (ignored)

kind=1 and kind=2 both carry ['requests', N, <field>] paths, so they share a
single path dispatcher.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int | None:
    """Ints only - bools and floats-as-durations are not useful here."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _extract_text(item: Any) -> str:
    """Extract plain text from a markdown content item or string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("value") or item.get("text") or ""
    return ""


def _uri_path(value: Any) -> str:
    """Pull a filesystem path out of a serialized VS Code URI object."""
    if isinstance(value, str):
        return value
    d = _as_dict(value)
    return d.get("fsPath") or d.get("path") or ""


def _epoch_ms(value: int) -> datetime:
    """Milliseconds to a UTC datetime, falling back to the epoch for values the
    platform cannot represent."""
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _dedupe(paths: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for p in paths:
        if p:
            seen.setdefault(p, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TodoItem:
    title: str
    status: str = ""


@dataclass
class SubagentCall:
    description: str = ""
    prompt: str = ""
    model_name: str = ""
    result: str = ""


@dataclass
class ToolCall:
    """A single tool invocation, merged from the response part and the
    matching entry in result.metadata.toolCallRounds."""
    tool_id: str = ""            # VS Code tool id, e.g. copilot_findFiles
    call_id: str = ""
    name: str = ""               # model-facing name, e.g. file_search
    title: str = ""              # generatedTitle
    message: str = ""            # invocationMessage
    past_tense: str = ""         # pastTenseMessage
    arguments: str = ""          # JSON argument string
    output: str = ""
    is_error: bool = False
    is_complete: bool = True
    # terminal-specific
    command: str = ""
    cwd: str = ""
    exit_code: int | None = None
    duration_ms: int | None = None
    # structured payloads
    todos: list[TodoItem] = field(default_factory=list)
    subagent: SubagentCall | None = None
    result_files: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Best available short name for the tool."""
        return self.name or self.tool_id or "tool"

    @property
    def summary(self) -> str:
        """Best available one-line description of what the call did."""
        return (self.title or self.message or self.past_tense or "").strip()


@dataclass
class ResponsePart:
    """A single rendered part of an assistant response."""
    kind: str          # 'text', 'thinking', 'tool_call', 'file_edit', 'warning'
    text: str = ""
    tool: ToolCall | None = None
    file_path: str = ""


@dataclass
class ChatRequest:
    request_id: str = ""
    timestamp_ms: int = 0
    model_id: str = ""
    user_text: str = ""
    is_system_initiated: bool = False
    system_label: str = ""
    response_parts: list[ResponsePart] = field(default_factory=list)
    # metrics
    completion_tokens: int | None = None
    input_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    elapsed_ms: int | None = None
    total_elapsed_ms: int | None = None
    first_progress_ms: int | None = None
    # outcome
    error_code: str = ""
    error_message: str = ""
    is_incomplete: bool = False
    stop_reason: str = ""
    # context
    mode_name: str = ""
    permission_level: str = ""
    agent_version: str = ""
    context_files: list[str] = field(default_factory=list)
    edited_files: list[str] = field(default_factory=list)

    @property
    def timestamp(self) -> datetime:
        return _epoch_ms(self.timestamp_ms)

    @property
    def duration_ms(self) -> int | None:
        """Wall-clock duration of the turn. elapsedMs is recorded for only
        about half of requests; result.timings.totalElapsed covers the rest
        and agrees with it to within ~100ms."""
        return self.elapsed_ms if self.elapsed_ms is not None else self.total_elapsed_ms

    @property
    def assistant_text(self) -> str:
        """Concatenated plain text parts of the response."""
        return "\n\n".join(
            p.text for p in self.response_parts if p.kind == "text" and p.text
        )

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [p.tool for p in self.response_parts if p.kind == "tool_call" and p.tool]

    @property
    def all_edited_files(self) -> list[str]:
        """Files touched this turn, from both edit events and inline edit parts."""
        inline = [p.file_path for p in self.response_parts if p.kind == "file_edit"]
        return _dedupe(self.edited_files + inline)


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
    responder: str = ""
    agent_version: str = ""
    cwd: str = ""
    git_branch: str = ""

    @property
    def created_at(self) -> datetime:
        return _epoch_ms(self.created_ms)

    @property
    def user_requests(self) -> list[ChatRequest]:
        """Only human-initiated requests with non-empty messages."""
        return [r for r in self.requests if not r.is_system_initiated and r.user_text]

    @property
    def models_used(self) -> list[tuple[str, int]]:
        counts = Counter(r.model_id for r in self.requests if r.model_id)
        return counts.most_common()

    @property
    def tool_counts(self) -> list[tuple[str, int]]:
        counts: Counter[str] = Counter()
        for r in self.requests:
            counts.update(t.label for t in r.tool_calls)
        return counts.most_common()

    @property
    def total_tool_calls(self) -> int:
        return sum(len(r.tool_calls) for r in self.requests)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens or 0 for r in self.requests)

    @property
    def peak_context_tokens(self) -> int:
        """Largest context a single turn was sent, fresh plus cached.

        Input and cache figures are a point-in-time context size, not a
        quantity produced, so they do not sum across turns the way completion
        tokens do. Providers also disagree on them: claude records one call's
        snapshot while codex and opencode accumulate a turn's calls. A maximum
        is the one reading that holds either way.
        """
        return max(
            ((r.input_tokens or 0) + (r.cache_read_tokens or 0) for r in self.requests),
            default=0,
        )

    @property
    def total_elapsed_ms(self) -> int:
        return sum(r.duration_ms or 0 for r in self.requests)

    @property
    def span_ms(self) -> int:
        """Elapsed time between the first and last turn, including idle gaps."""
        stamps = [r.timestamp_ms for r in self.requests if r.timestamp_ms]
        if len(stamps) < 2:
            return 0
        last = max(stamps)
        tail = next((r for r in self.requests if r.timestamp_ms == last), None)
        return last - min(stamps) + ((tail.duration_ms or 0) if tail else 0)

    @property
    def edited_files(self) -> list[str]:
        out: list[str] = []
        for r in self.requests:
            out.extend(r.all_edited_files)
        return _dedupe(out)


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

def _join_output(output: Any) -> str:
    """Flatten resultDetails.output, a list of {type, isText, value} entries."""
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        return ""
    chunks: list[str] = []
    for entry in output:
        if isinstance(entry, str):
            chunks.append(entry)
        elif isinstance(entry, dict):
            text = _extract_text(entry)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _fill_terminal(tool: ToolCall, specific: dict[str, Any]) -> None:
    command = specific.get("commandLine")
    if isinstance(command, dict):
        # toolEdited is what actually ran when the tool rewrote the command
        tool.command = (command.get("toolEdited") or command.get("original") or "").strip()
    else:
        tool.command = str(command or "").strip()

    tool.cwd = _uri_path(specific.get("cwd"))

    state = _as_dict(specific.get("terminalCommandState"))
    tool.exit_code = _as_int(state.get("exitCode"))
    tool.duration_ms = _as_int(state.get("duration"))

    text = _extract_text(specific.get("terminalCommandOutput"))
    if text:
        tool.output = text.replace("\r\n", "\n")

    if tool.exit_code not in (None, 0):
        tool.is_error = True


def _parse_tool_call(item: dict[str, Any]) -> ToolCall:
    tool = ToolCall(
        tool_id=item.get("toolId", ""),
        call_id=item.get("toolCallId", ""),
        title=_extract_text(item.get("generatedTitle")),
        message=_extract_text(item.get("invocationMessage")),
        past_tense=_extract_text(item.get("pastTenseMessage")),
        is_complete=bool(item.get("isComplete", True)),
    )

    details = item.get("resultDetails")
    if isinstance(details, dict):
        tool.arguments = details.get("input") or ""
        tool.output = _join_output(details.get("output"))
        tool.is_error = bool(details.get("isError"))
    elif isinstance(details, list):
        # Some tools report results as a plain list of file URIs
        tool.result_files = _dedupe([_uri_path(x) for x in details])

    specific = _as_dict(item.get("toolSpecificData"))
    kind = specific.get("kind")
    if kind == "terminal":
        _fill_terminal(tool, specific)
    elif kind == "todoList":
        raw_todos = specific.get("todoList")
        if isinstance(raw_todos, list):
            tool.todos = [
                TodoItem(title=str(t.get("title", "")), status=str(t.get("status", "")))
                for t in raw_todos
                if isinstance(t, dict)
            ]
    elif kind == "subagent":
        tool.subagent = SubagentCall(
            description=str(specific.get("description") or ""),
            prompt=str(specific.get("prompt") or ""),
            model_name=str(specific.get("modelName") or ""),
            result=str(specific.get("result") or ""),
        )
    elif kind == "input" and not tool.arguments:
        raw_input = specific.get("rawInput")
        if raw_input:
            tool.arguments = (
                json.dumps(raw_input, indent=2)
                if isinstance(raw_input, (dict, list))
                else str(raw_input)
            )

    return tool


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
            parts.append(ResponsePart(kind="tool_call", tool=_parse_tool_call(item)))

        elif kind == "textEditGroup":
            path = _uri_path(item.get("uri"))
            if path:
                parts.append(ResponsePart(kind="file_edit", file_path=path))

        elif kind == "warning":
            text = _extract_text(item)
            if text:
                parts.append(ResponsePart(kind="warning", text=text))

        # Remaining kinds (inlineReference, undoStop, codeblockUri,
        # mcpServersStarting, progressTaskSerialized, questionCarousel) are
        # structural/UI items with nothing to export.

    return parts


def _apply_tool_arguments(req: ChatRequest, result: dict[str, Any]) -> None:
    """Merge tool names and argument JSON from result.metadata.toolCallRounds.

    Round ids carry a '__vscode-<n>' suffix that the response part's
    toolCallId does not, so match on the prefix.
    """
    metadata = _as_dict(result.get("metadata"))
    rounds = metadata.get("toolCallRounds")
    if not isinstance(rounds, list):
        return

    by_id: dict[str, dict[str, Any]] = {}
    for rnd in rounds:
        for call in _as_dict(rnd).get("toolCalls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "").split("__vscode-")[0]
            if call_id:
                by_id[call_id] = call

    for tool in req.tool_calls:
        call = by_id.get(tool.call_id)
        if not call:
            continue
        tool.name = str(call.get("name") or "")
        if not tool.arguments:
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    tool.arguments = json.dumps(json.loads(args), indent=2)
                except (json.JSONDecodeError, TypeError):
                    tool.arguments = args
            elif isinstance(args, (dict, list)):
                tool.arguments = json.dumps(args, indent=2)


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------

def _build_request(raw: dict[str, Any]) -> ChatRequest:
    msg = _as_dict(raw.get("message"))

    req = ChatRequest(
        request_id=raw.get("requestId", ""),
        timestamp_ms=_as_int(raw.get("timestamp")) or 0,
        model_id=raw.get("modelId", ""),
        user_text=msg.get("text", ""),
        is_system_initiated=bool(raw.get("isSystemInitiated", False)),
        system_label=_extract_text(raw.get("systemInitiatedLabel")),
    )

    mode = _as_dict(raw.get("modeInfo"))
    req.mode_name = str(mode.get("modeName") or "")
    req.permission_level = str(mode.get("permissionLevel") or "")
    req.agent_version = str(_as_dict(raw.get("agent")).get("extensionVersion") or "")

    req.completion_tokens = _as_int(raw.get("completionTokens"))
    req.elapsed_ms = _as_int(raw.get("elapsedMs"))

    result = _as_dict(raw.get("result"))
    timings = _as_dict(result.get("timings"))
    req.total_elapsed_ms = _as_int(timings.get("totalElapsed"))
    req.first_progress_ms = _as_int(timings.get("firstProgress"))

    error = _as_dict(result.get("errorDetails"))
    req.error_code = str(error.get("code") or "")
    req.error_message = str(error.get("message") or "")
    req.is_incomplete = bool(error.get("responseIsIncomplete"))

    refs = raw.get("contentReferences")
    if isinstance(refs, list):
        req.context_files = _dedupe(
            [_uri_path(_as_dict(r).get("reference")) for r in refs if isinstance(r, dict)]
        )

    events = raw.get("editedFileEvents")
    if isinstance(events, list):
        req.edited_files = _dedupe(
            [_uri_path(_as_dict(e).get("uri")) for e in events if isinstance(e, dict)]
        )

    raw_resp = raw.get("response")
    if isinstance(raw_resp, list):
        req.response_parts = _parse_response(raw_resp)

    _apply_tool_arguments(req, result)
    return req


# ---------------------------------------------------------------------------
# JSONL reconstruction
# ---------------------------------------------------------------------------

def _apply_update(
    state: dict[str, Any],
    raw_requests: list[dict[str, Any]],
    keys: list[Any],
    value: Any,
) -> None:
    """Apply one kind=1/kind=2 update to the reconstructed session state."""
    if not keys:
        return

    if keys == ["requests"]:
        if isinstance(value, list):
            if len(value) == 1:
                # Single-item append pattern
                raw_requests.append(value[0])
            else:
                # Full replacement (rare)
                raw_requests[:] = list(value)
        return

    if (
        len(keys) == 3
        and keys[0] == "requests"
        and isinstance(keys[1], int)
        and isinstance(keys[2], str)
    ):
        idx: int = keys[1]
        # Grow array if needed (fields may arrive before the request itself)
        while len(raw_requests) <= idx:
            raw_requests.append({})
        raw_requests[idx][keys[2]] = value
        return

    if len(keys) == 1 and isinstance(keys[0], str):
        state[keys[0]] = value


# ---------------------------------------------------------------------------
# Public parse function
# ---------------------------------------------------------------------------

def parse_session(path: Path) -> ChatSession | None:
    """Parse a .jsonl chat session file and return a ChatSession, or None if empty."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    state: dict[str, Any] = {}
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

        elif kind in (1, 2):
            _apply_update(state, raw_requests, record.get("k") or [], record.get("v"))

    if not raw_requests:
        return None

    requests = [_build_request(r) for r in raw_requests if isinstance(r, dict) and r]
    if not requests:
        return None

    session_id = state.get("sessionId", path.stem)
    title = state.get("customTitle", "").strip() or _derive_title(raw_requests)
    created_ms = int(state.get("creationDate") or 0)
    model_id = _as_dict(_as_dict(state.get("inputState")).get("selectedModel")).get(
        "identifier", ""
    )

    return ChatSession(
        session_id=session_id,
        title=title or session_id,
        created_ms=created_ms,
        model_id=model_id,
        requests=requests,
        responder=str(state.get("responderUsername") or ""),
        agent_version=next((r.agent_version for r in requests if r.agent_version), ""),
    )


def _derive_title(raw_requests: list[dict[str, Any]], max_len: int = 60) -> str:
    """Derive a title from the first user message when no customTitle exists."""
    for r in raw_requests:
        if not isinstance(r, dict) or r.get("isSystemInitiated"):
            continue
        text = _as_dict(r.get("message")).get("text", "").strip()
        if text:
            # Strip markdown and take first line
            first_line = text.splitlines()[0].strip().lstrip("#").strip()
            return first_line[:max_len] + ("..." if len(first_line) > max_len else "")
    return "Untitled Session"
