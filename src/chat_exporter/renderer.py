"""
Render ChatSession objects to Markdown (or JSON) strings.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import asdict
from pathlib import Path

from .parser import ChatRequest, ChatSession, ResponsePart, ToolCall
from .prompts import display_prompt


# ---------------------------------------------------------------------------
# Rendering options
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOOL_OUTPUT = 2000
DEFAULT_MAX_ASSISTANT_CHARS = 2000

# Tools whose output is a wall of file content rather than a result worth
# reading back; their invocation line already says which file was read.
_QUIET_TOOLS = {"read_file", "copilot_readFile"}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|]')
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_MD_ESCAPE_RE = re.compile(r"\\([_*`\[\]#])")
_DETAILS_RE = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)

# Stop reasons that mean the turn ended the way it was meant to; anything else
# is worth showing next to the turn. Spellings vary by provider.
_NORMAL_STOPS = {
    "end_turn", "tool_use", "stop_sequence", "pause_turn", "stop", "tool-calls", "tool_calls",
}

# What the agent_version on a session is the version of.
_CLIENT_LABELS = {
    "copilot": "Copilot Chat",
    "claude": "Claude Code",
    "codex": "Codex CLI",
    "opencode": "opencode",
}

_STATUS_MARKS = {
    "completed": "x",
    "in-progress": "~",
    "not-started": " ",
}


def safe_filename(name: str, max_len: int = 80) -> str:
    """Convert a string to a safe cross-platform filename (no extension)."""
    name = _UNSAFE_FILENAME_RE.sub("-", name)
    name = _MULTI_SPACE_RE.sub(" ", name).strip(". ")
    return name[:max_len] if name else "unnamed"


def _fence(text: str, lang: str = "") -> str:
    """Wrap text in a code fence long enough to survive backticks inside it."""
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text}\n{ticks}"


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    kept = text[:limit].rstrip()
    return f"{kept}\n... [{len(text) - len(kept):,} more characters truncated]"


def _bound_middle(compact: str, limit: int) -> str:
    """Keep the start and end of `compact` when it exceeds `limit`."""
    if limit <= 0 or len(compact) <= limit:
        return compact
    marker = " [assistant response condensed] "
    if limit <= len(marker):
        return marker[:limit]
    head = max(1, (limit - len(marker)) // 2)
    tail = limit - len(marker) - head
    return compact[:head].rstrip() + marker + compact[-tail:].lstrip()


def _condense_assistant_text(text: str, limit: int) -> str:
    """Keep readable assistant prose while dropping embedded detail blocks."""
    text = _DETAILS_RE.sub("", text)
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and line.strip():
            lines.append(line.strip())
    compact = _MULTI_SPACE_RE.sub(" ", " ".join(lines)).strip()
    return _bound_middle(compact, limit)


def _prose_paragraphs(text: str) -> list[str]:
    """Assistant prose as paragraphs (blank-line separated), without detail
    blocks and code fences; each paragraph is one whitespace-collapsed line."""
    text = _DETAILS_RE.sub("", text)
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(_MULTI_SPACE_RE.sub(" ", " ".join(current)).strip())
            current = []
    if current:
        paragraphs.append(_MULTI_SPACE_RE.sub(" ", " ".join(current)).strip())
    return paragraphs


def _collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines, but never inside a code fence."""
    out: list[str] = []
    blanks = 0
    in_fence = False
    fence_marker = ""

    for line in text.split("\n"):
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                in_fence, fence_marker = False, ""

        if in_fence or line.strip():
            blanks = 0
            out.append(line)
        else:
            blanks += 1
            if blanks < 2:
                out.append(line)

    return "\n".join(out)


def _balance_fences(text: str) -> str:
    """Close a code fence the text leaves open.

    Thinking blocks go into <details> as raw markdown so their own formatting
    survives; a stray opening fence would otherwise swallow the closing tag
    and every turn after it.
    """
    marker = ""
    for line in text.split("\n"):
        match = _FENCE_RE.match(line)
        if not match:
            continue
        token = match.group(1)
        if not marker:
            marker = token
        elif token[0] == marker[0] and len(token) >= len(marker):
            marker = ""
    return f"{text}\n{marker}" if marker else text


def _short_path(path: str) -> str:
    """Shorten an absolute path against the user's home directory."""
    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except (ValueError, RuntimeError):
        return path


def _fmt_duration(ms: int | None) -> str:
    if not ms or ms < 0:
        return ""
    if ms < 1000:
        return f"{ms}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _fmt_tokens(req: ChatRequest) -> str:
    """One token figure per turn, or a breakdown when the provider records more
    than the completion count."""
    if not req.completion_tokens and not req.input_tokens:
        return ""
    context = [
        (req.input_tokens, "in"),
        (req.cache_read_tokens, "cached"),
        (req.reasoning_tokens, "reasoning"),
    ]
    shown = [f"{count:,} {label}" for count, label in context if count]
    if not shown:
        return f"{req.completion_tokens:,} tokens"
    return " / ".join([f"{req.completion_tokens or 0:,} out", *shown]) + " tokens"


def _fmt_ts(session: ChatSession) -> str:
    dt = session.created_at.astimezone()
    return dt.strftime("%Y-%m-%d %H:%M %Z")


def _join_meta(parts: list[str]) -> str:
    return " · ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Tool call rendering
# ---------------------------------------------------------------------------

def _link_label(url: str) -> str:
    """Human-readable label for a bare markdown link target."""
    tail = url.split("#")[0].rstrip("/").rsplit("/", 1)[-1]
    return tail or url


def _clean_summary(text: str) -> str:
    """Flatten a Copilot invocation message into one plain line.

    These carry markdown links (often with empty labels) and backslash
    escapes, neither of which renders inside an HTML <summary>.
    """
    text = text.replace("\n", " ")
    text = _MD_LINK_RE.sub(lambda m: m.group(1).strip() or _link_label(m.group(2)), text)
    text = _MD_ESCAPE_RE.sub(r"\1", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def _ellipsize(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def _tool_headline(tool: ToolCall) -> str:
    # For terminal calls the command itself is the most informative label.
    subject = _clean_summary(tool.command or tool.summary)

    parts = [f"<b>{html.escape(tool.label)}</b>"]
    if subject:
        parts.append(f"<code>{html.escape(_ellipsize(subject, 90))}</code>")

    status: list[str] = []
    if tool.exit_code is not None:
        status.append(f"exit {tool.exit_code}")
    elif tool.is_error:
        status.append("error")
    if not tool.is_complete:
        status.append("incomplete")
    duration = _fmt_duration(tool.duration_ms)
    if duration:
        status.append(duration)

    headline = " \u2014 ".join(parts)
    if status:
        headline += " \u00b7 " + " \u00b7 ".join(status)
    return ("\u26a0 " if tool.is_error else "") + headline


def _render_tool_call(
    tool: ToolCall,
    max_output: int,
    include_args: bool,
) -> str:
    lines = [f"<details><summary>{_tool_headline(tool)}</summary>", ""]

    if tool.command:
        lines += [_fence(tool.command, "sh"), ""]
        if tool.cwd:
            lines += [f"_cwd: `{_short_path(tool.cwd)}`_", ""]

    if include_args and tool.arguments and not tool.command:
        # Not every tool takes JSON: the CLI's apply_patch takes a raw patch.
        body = tool.arguments.strip()
        try:
            json.loads(body)
            language = "json"
        except (json.JSONDecodeError, ValueError):
            language = "diff" if body.startswith("*** ") else ""
        lines += ["**Arguments:**", "", _fence(body, language), ""]

    if tool.todos:
        lines.append("**Todo list:**")
        lines.append("")
        for todo in tool.todos:
            mark = _STATUS_MARKS.get(todo.status, " ")
            lines.append(f"- [{mark}] {todo.title}")
        lines.append("")

    if tool.subagent:
        sub = tool.subagent
        lines.append("**Subagent:** " + (sub.description or "(no description)") + "  ")
        if sub.model_name:
            lines.append(f"**Subagent model:** `{sub.model_name}`  ")
        lines.append("")
        if sub.prompt:
            lines += ["**Prompt:**", "", _fence(_truncate(sub.prompt.strip(), max_output)), ""]
        if sub.result:
            lines += ["**Result:**", "", _fence(_truncate(sub.result.strip(), max_output)), ""]

    if tool.result_files:
        lines.append("**Files returned:**")
        lines.append("")
        for path in tool.result_files:
            lines.append(f"- `{_short_path(path)}`")
        lines.append("")

    output = tool.output.strip()
    if output and tool.label not in _QUIET_TOOLS:
        header = "**Output:**"
        if tool.exit_code not in (None, 0):
            header = f"**Output** (exit {tool.exit_code}):"
        elif tool.is_error:
            header = "**Output** (error):"
        lines += [header, "", _fence(_truncate(output, max_output)), ""]

    lines.append("</details>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Turn rendering
# ---------------------------------------------------------------------------

def _assistant_meta(req: ChatRequest, include_metrics: bool) -> str:
    if not include_metrics:
        return ""
    bits: list[str] = []
    if req.model_id:
        # `copilot/auto` names a router, not a model; the metadata records what
        # it actually picked, so show that rather than leave the turn ambiguous.
        resolved = req.resolved_model
        if resolved and resolved not in req.model_id:
            bits.append(f"`{req.model_id}` -> `{resolved}`")
        else:
            bits.append(f"`{req.model_id}`")
    elif req.resolved_model:
        bits.append(f"`{req.resolved_model}`")
    duration = _fmt_duration(req.duration_ms)
    if duration:
        bits.append(duration)
    first = _fmt_duration(req.first_progress_ms)
    if first:
        bits.append(f"first token {first}")
    tokens = _fmt_tokens(req)
    if tokens:
        bits.append(tokens)
    if req.stop_reason and req.stop_reason not in _NORMAL_STOPS:
        bits.append(f"stopped: {req.stop_reason}")
    if req.permission_level and req.permission_level != "default":
        bits.append(req.permission_level)
    return _join_meta(bits)


def _render_response(
    req: ChatRequest,
    include_tools: bool,
    include_thinking: bool,
    max_output: int,
    include_args: bool,
) -> str:
    sections: list[str] = []

    for part in req.response_parts:
        if part.kind == "text" and part.text.strip():
            sections.append(part.text.strip())
        elif part.kind == "thinking" and include_thinking and part.text.strip():
            sections.append(
                "<details><summary>Thinking</summary>\n\n"
                + _balance_fences(part.text.strip())
                + "\n\n</details>"
            )
        elif part.kind == "tool_call" and include_tools and part.tool:
            sections.append(_render_tool_call(part.tool, max_output, include_args))
        elif part.kind == "warning" and part.text.strip():
            sections.append(f"> **Warning:** {part.text.strip()}")

    return "\n\n".join(sections)


def _render_request(
    req: ChatRequest,
    index: int,
    include_tools: bool,
    include_thinking: bool,
    include_metrics: bool,
    include_files: bool,
    include_context: bool,
    max_output: int,
    include_args: bool,
    turn_header: bool = True,
    notification_lines: bool = False,
) -> str:
    """Render one request. A background-task notification is shown as its
    one-line form, never as a User prompt; with turn_header=False it continues
    the turn before it."""
    lines: list[str] = ["---", ""] if turn_header else []

    ts = req.timestamp.astimezone().strftime("%H:%M:%S")
    if turn_header:
        lines.append(f"### Turn {index} \u00b7 {ts}")
        lines.append("")
    if req.is_notification and notification_lines:
        lines.append(f"**Background task** ({ts})")
        lines.append("")
        lines.append(_notification_line(req))
    else:
        lines.append("**User**")
        lines.append("")
        lines.append(req.user_text.strip())
    lines.append("")

    if include_context and req.context_files:
        shown = ", ".join(f"`{_short_path(p)}`" for p in req.context_files[:12])
        extra = len(req.context_files) - 12
        lines.append(f"_Context: {shown}" + (f" +{extra} more" if extra > 0 else "") + "_")
        lines.append("")

    # Assistant turn
    body = _render_response(
        req,
        include_tools=include_tools,
        include_thinking=include_thinking,
        max_output=max_output,
        include_args=include_args,
    )
    meta = _assistant_meta(req, include_metrics)
    if body or meta:
        lines.append("**Assistant**" + (f" \u2014 {meta}" if meta else ""))
        lines.append("")
        if body:
            lines.append(body)
            lines.append("")

    if req.error_code or req.is_incomplete:
        # Error messages may span lines; a blockquote needs them on one.
        detail = _MULTI_SPACE_RE.sub(
            " ", (req.error_message or req.error_code or "response incomplete").replace("\n", " ")
        ).strip()
        label = "Interrupted" if req.error_code == "canceled" else "Error"
        lines.append(f"> **{label}:** {detail} (`{req.error_code or 'incomplete'}`)")
        lines.append("")

    if include_files:
        edited = req.all_edited_files
        if edited:
            shown = ", ".join(f"`{_short_path(p)}`" for p in edited[:12])
            extra = len(edited) - 12
            lines.append(
                f"_Files changed: {shown}" + (f" +{extra} more" if extra > 0 else "") + "_"
            )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Condensed and standard detail levels
# ---------------------------------------------------------------------------

DETAIL_LEVELS = ("condensed", "standard", "full")
MIN_REPEAT_CHARS = 40
_SUBAGENT_TOOLS = {"agent", "task"}


def _clip(text: str, limit: int) -> str:
    text = _MULTI_SPACE_RE.sub(" ", " ".join(text.split()))
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _note_field(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return " ".join(match.group(1).split()) if match else ""


def _notification_line(req: ChatRequest) -> str:
    """One line for a Claude Code background-task notification."""
    status = _note_field(req.user_text, "status") or "finished"
    summary = _note_field(req.user_text, "summary") or "(no summary)"
    result = _note_field(req.user_text, "result")
    line = f"- [background task {status}] {_clip(summary, 160)}"
    return line + (f" -- {_clip(result, 200)}" if result else "")


def _is_subagent(tool: ToolCall) -> bool:
    return tool.subagent is not None or tool.label.casefold() in _SUBAGENT_TOOLS


def _arg_value(tool: ToolCall, *keys: str) -> str:
    try:
        data = json.loads(tool.arguments) if tool.arguments else {}
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _tool_state(tool: ToolCall) -> str:
    if tool.is_error or tool.exit_code not in (None, 0):
        return "error"
    return "ok" if tool.is_complete else "incomplete"


def _subagent_line(tool: ToolCall) -> str:
    sub = tool.subagent
    description = (
        (sub.description if sub else "")
        or _arg_value(tool, "description", "name", "agent_type", "subagent_type")
        or tool.summary
        or tool.label
    )
    result = (sub.result if sub and sub.result else "") or tool.output
    line = f"- [subagent {_tool_state(tool)}] {_clip(description, 160)}"
    return line + (f" -- {_clip(result, 200)}" if result.strip() else "")


def _tool_line(tool: ToolCall) -> str:
    preview = tool.command or tool.summary or tool.arguments
    line = f"- tool {tool.label}"
    if preview.strip():
        line += f": {_clip(preview, 120)}"
    return f"{line} [{_tool_state(tool)}]"


def _group_turns(requests: list[ChatRequest]) -> list[list[ChatRequest]]:
    """Group requests into turns; a notification joins the turn before it."""
    groups: list[list[ChatRequest]] = []
    for req in requests:
        if req.is_notification and groups:
            groups[-1].append(req)
        else:
            groups.append([req])
    return groups


def _error_line(req: ChatRequest) -> str:
    detail = _MULTI_SPACE_RE.sub(
        " ", (req.error_message or req.error_code or "response incomplete").replace("\n", " ")
    ).strip()
    label = "Interrupted" if req.error_code == "canceled" else "Error"
    return f"> **{label}:** {detail}"


def _summary_blocks(
    requests: list[ChatRequest], detail: str, max_assistant_chars: int
) -> list[str]:
    """One Markdown block per turn for the condensed and standard levels."""
    seen: dict[str, int] = {}
    blocks: list[str] = []
    for number, group in enumerate(_group_turns(requests), start=1):
        # (kind, text, new): kind is prose, line or error; for prose, `new`
        # lists (key, paragraph) pairs that are first copies.
        segments: list[tuple[str, str, list[tuple[str, str]]]] = []
        for req in group:
            if req.is_notification:
                segments.append(("line", _notification_line(req), []))
            for part in req.response_parts:
                if part.kind == "text" and part.text.strip():
                    prose: list[str] = []
                    new: list[tuple[str, str]] = []
                    for paragraph in _prose_paragraphs(part.text):
                        key = paragraph.casefold()
                        if len(key) >= MIN_REPEAT_CHARS and key in seen:
                            prose.append(f"[repeated from turn {seen[key]}]")
                        else:
                            if len(key) >= MIN_REPEAT_CHARS:
                                new.append((key, paragraph))
                            prose.append(paragraph)
                    if prose:
                        segments.append(("prose", "\n\n".join(prose), new))
                elif part.kind == "tool_call" and part.tool:
                    if _is_subagent(part.tool):
                        segments.append(("line", _subagent_line(part.tool), []))
                    elif detail == "standard":
                        segments.append(("line", _tool_line(part.tool), []))
            if req.error_code or req.is_incomplete:
                segments.append(("error", _error_line(req), []))

        # The prose limit applies per turn, as in the old condensed view.
        prose_count = sum(k == "prose" for k, _, _ in segments)
        if max_assistant_chars > 0 and sum(len(t) for k, t, _ in segments if k == "prose") > max_assistant_chars:
            share = max(1, max_assistant_chars // max(1, prose_count))
            segments = [
                (k, _bound_middle(t, share), n) if k == "prose" else (k, t, n)
                for k, t, n in segments
            ]
        # Only a paragraph that is shown whole can be pointed to later.
        for kind, text, new in segments:
            if kind == "prose":
                for key, paragraph in new:
                    if paragraph in text:
                        seen.setdefault(key, number)

        if group[0].is_notification:
            lead = ["**Background task**", "", _notification_line(group[0]), ""]
            segments = [seg for seg in segments if not (seg[0] == "line" and seg[1] == _notification_line(group[0]))]
        else:
            lead = ["**User**", "", display_prompt(group[0].user_text), ""]
        lines = ["---", "", f"### Turn {number}", ""] + lead
        body: list[str] = []
        previous = ""
        for kind, text, _ in segments:
            if body:
                body.append("\n" if kind == "line" and previous == "line" else "\n\n")
            body.append(text)
            previous = kind
        if body:
            lines += ["**Assistant**", "", "".join(body), ""]
        blocks.append("\n".join(lines))
    return blocks


# ---------------------------------------------------------------------------
# Session header
# ---------------------------------------------------------------------------

def _render_header(session: ChatSession, include_metrics: bool) -> list[str]:
    lines = [f"# {session.title}", ""]
    lines.append(f"**Session ID:** `{session.session_id}`  ")
    lines.append(f"**Provider:** `{session.provider}`  ")
    lines.append(f"**Workspace:** `{session.workspace}`  ")
    lines.append(f"**Created:** {_fmt_ts(session)}  ")

    turns = len(session.user_requests)
    span = _fmt_duration(session.span_ms)
    lines.append(f"**Turns:** {turns}" + (f"  \u00b7  **Span:** {span}" if span else "") + "  ")

    models = session.models_used
    if models:
        rendered = ", ".join(
            f"`{name}`" + (f" ({count})" if len(models) > 1 else "")
            for name, count in models
        )
        lines.append(f"**Models:** {rendered}  ")
    elif session.model_id:
        lines.append(f"**Model:** `{session.model_id}`  ")

    if include_metrics:
        active = _fmt_duration(session.total_elapsed_ms)
        if active:
            lines.append(f"**Time generating:** {active}  ")
        tokens = session.total_completion_tokens
        if tokens:
            lines.append(f"**Completion tokens:** {tokens:,}  ")
        peak = session.peak_context_tokens
        if peak:
            lines.append(f"**Peak context:** {peak:,} tokens  ")

        tools = session.tool_counts
        if tools:
            top = ", ".join(f"{name} ({count})" for name, count in tools[:6])
            extra = len(tools) - 6
            lines.append(
                f"**Tool calls:** {session.total_tool_calls:,} \u2014 {top}"
                + (f", +{extra} more" if extra > 0 else "")
                + "  "
            )

    if session.git_branch:
        lines.append(f"**Branch:** `{session.git_branch}`  ")

    if session.agent_version:
        label = _CLIENT_LABELS.get(session.provider, "Client")
        lines.append(f"**{label}:** `{session.agent_version}`  ")

    lines.append("")
    return lines


def _render_condensed_header(session: ChatSession) -> list[str]:
    stamp = session.last_activity_at
    last = stamp.astimezone().strftime("%Y-%m-%d %H:%M %Z") if stamp else "unknown"
    return [
        f"# {session.title}",
        "",
        f"**Session:** `{session.session_id}` \u00b7 **Provider:** `{session.provider}` \u00b7 "
        f"**Workspace:** `{session.workspace}`",
        f"**Turns:** {session.turn_count} \u00b7 **Last activity:** {last}",
        "",
    ]


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------

def resolve_detail(
    detail: str | None = None, condensed: bool | None = None, default: str = "condensed"
) -> str:
    """Pick the detail level. An explicit `detail` wins; otherwise the legacy
    `condensed` flag maps True to "condensed" and False to "full"; with neither,
    the level is `default`."""
    if detail is not None:
        if detail not in DETAIL_LEVELS:
            raise ValueError(f"detail must be one of {', '.join(DETAIL_LEVELS)}")
        return detail
    if condensed is None:
        return default
    return "condensed" if condensed else "full"


def render_session_parts(
    session: ChatSession,
    include_tools: bool = True,
    include_thinking: bool = False,
    include_metrics: bool = True,
    include_files: bool = True,
    include_context: bool = False,
    include_tool_args: bool = False,
    max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT,
    condensed: bool | None = None,
    max_assistant_chars: int = DEFAULT_MAX_ASSISTANT_CHARS,
    detail: str | None = None,
    group_notifications: bool = False,
) -> tuple[str, list[str], str]:
    """Render a session as (header, one block per turn, footer).

    Joining the three with newlines gives the whole transcript; callers that
    page a long transcript cut between blocks so no turn is split.
    """
    level = resolve_detail(detail, condensed, default="full")
    if level == "full":
        header = "\n".join(_render_header(session, include_metrics))
    else:
        header = "\n".join(_render_condensed_header(session))

    user_reqs = session.user_requests
    if not user_reqs:
        return header, [], "_No messages in this session._"

    if level != "full":
        return header, _summary_blocks(user_reqs, level, max(0, max_assistant_chars)), ""

    def render(req: ChatRequest, index: int, turn_header: bool = True) -> str:
        return _render_request(
            req,
            index,
            include_tools=include_tools,
            include_thinking=include_thinking,
            include_metrics=include_metrics,
            include_files=include_files,
            include_context=include_context,
            max_output=max_tool_output,
            include_args=include_tool_args,
            turn_header=turn_header,
            notification_lines=group_notifications,
        )

    if group_notifications:
        # Same turn numbering as the condensed levels: a notification joins the
        # turn before it.
        blocks = [
            "\n".join(
                render(req, number, turn_header=position == 0)
                for position, req in enumerate(group)
            )
            for number, group in enumerate(_group_turns(user_reqs), start=1)
        ]
    else:
        blocks = [render(req, idx) for idx, req in enumerate(user_reqs, start=1)]
    footer = ""
    edited = session.edited_files if include_files else []
    if edited:
        footer = "\n".join(
            ["---", "", f"## Files changed ({len(edited)})", ""]
            + [f"- `{_short_path(p)}`" for p in edited]
            + [""]
        )
    return header, blocks, footer


def render_session(
    session: ChatSession,
    include_tools: bool = True,
    include_thinking: bool = False,
    include_metrics: bool = True,
    include_files: bool = True,
    include_context: bool = False,
    include_tool_args: bool = False,
    max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT,
    condensed: bool | None = None,
    max_assistant_chars: int = DEFAULT_MAX_ASSISTANT_CHARS,
    detail: str | None = None,
    group_notifications: bool = False,
) -> str:
    """Render a ChatSession as Markdown.

    Detail levels: "condensed" keeps user prompts and assistant prose plus one
    line per subagent or background-task result; "standard" adds one line per
    tool call; "full" renders tool calls with output, metrics and file lists.
    The legacy `condensed` flag maps True to "condensed" and False to "full".
    Assistant prose repeated from an earlier paragraph is replaced by a
    pointer in condensed and standard output. With group_notifications, "full"
    numbers turns the way the other levels do (notifications join the turn before
    them and show as a labeled line); the export CLI leaves it off.
    """
    header, blocks, footer = render_session_parts(
        session, include_tools, include_thinking, include_metrics, include_files,
        include_context, include_tool_args, max_tool_output, condensed,
        max_assistant_chars, detail, group_notifications,
    )
    parts = [header, *blocks] + ([footer] if footer else [])
    return _collapse_blank_lines("\n".join(parts)).rstrip() + "\n"


def render_session_json(
    session: ChatSession,
    include_thinking: bool = False,
    max_tool_output: int = 0,
) -> str:
    """Render a ChatSession as a JSON document (full structured detail)."""
    data = asdict(session)

    data["created_at"] = session.created_at.isoformat()
    data["span_ms"] = session.span_ms
    data["total_elapsed_ms"] = session.total_elapsed_ms
    data["total_completion_tokens"] = session.total_completion_tokens
    data["peak_context_tokens"] = session.peak_context_tokens
    data["total_tool_calls"] = session.total_tool_calls
    data["models_used"] = dict(session.models_used)
    data["tool_counts"] = dict(session.tool_counts)
    data["edited_files"] = session.edited_files

    for raw_req, req in zip(data["requests"], session.requests):
        raw_req["timestamp"] = req.timestamp.isoformat() if req.timestamp_ms else None
        raw_req["duration_ms"] = req.duration_ms
        parts = []
        for raw_part, part in zip(raw_req["response_parts"], req.response_parts):
            if part.kind == "thinking" and not include_thinking:
                continue
            if max_tool_output and raw_part.get("tool"):
                tool = raw_part["tool"]
                tool["output"] = _truncate(tool.get("output") or "", max_tool_output)
            parts.append(raw_part)
        raw_req["response_parts"] = parts

    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
