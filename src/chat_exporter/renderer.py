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


# ---------------------------------------------------------------------------
# Rendering options
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOOL_OUTPUT = 2000

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
        lines += ["**Arguments:**", "", _fence(tool.arguments.strip(), "json"), ""]

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
        bits.append(f"`{req.model_id}`")
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
) -> str:
    lines: list[str] = ["---", ""]

    # User turn
    ts = req.timestamp.astimezone().strftime("%H:%M:%S")
    lines.append(f"### Turn {index} \u00b7 {ts}")
    lines.append("")
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
            context = [
                (session.total_input_tokens, "input"),
                (session.total_cache_read_tokens, "cached"),
            ]
            extra = ", ".join(f"{count:,} {label}" for count, label in context if count)
            lines.append(
                f"**Completion tokens:** {tokens:,}" + (f"  ({extra})" if extra else "") + "  "
            )

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


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------

def render_session(
    session: ChatSession,
    include_tools: bool = True,
    include_thinking: bool = False,
    include_metrics: bool = True,
    include_files: bool = True,
    include_context: bool = False,
    include_tool_args: bool = False,
    max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT,
) -> str:
    """Render a ChatSession as a Markdown document string."""
    lines = _render_header(session, include_metrics)

    user_reqs = session.user_requests
    if not user_reqs:
        lines.append("_No messages in this session._")
        return "\n".join(lines)

    for idx, req in enumerate(user_reqs, start=1):
        lines.append(
            _render_request(
                req,
                idx,
                include_tools=include_tools,
                include_thinking=include_thinking,
                include_metrics=include_metrics,
                include_files=include_files,
                include_context=include_context,
                max_output=max_tool_output,
                include_args=include_tool_args,
            )
        )

    if include_files:
        edited = session.edited_files
        if edited:
            lines.append("---")
            lines.append("")
            lines.append(f"## Files changed ({len(edited)})")
            lines.append("")
            lines.extend(f"- `{_short_path(p)}`" for p in edited)
            lines.append("")

    return _collapse_blank_lines("\n".join(lines)).rstrip() + "\n"


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
    data["total_input_tokens"] = session.total_input_tokens
    data["total_cache_read_tokens"] = session.total_cache_read_tokens
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
