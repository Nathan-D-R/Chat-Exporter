"""
Render ChatSession objects to Markdown strings.
"""

from __future__ import annotations

import re
from datetime import timezone

from .parser import ChatRequest, ChatSession, ResponsePart


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|]')
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def safe_filename(name: str, max_len: int = 80) -> str:
    """Convert a string to a safe cross-platform filename (no extension)."""
    name = _UNSAFE_FILENAME_RE.sub("-", name)
    name = _MULTI_SPACE_RE.sub(" ", name).strip(". ")
    return name[:max_len] if name else "unnamed"


def _fmt_ts(session: ChatSession) -> str:
    dt = session.created_at.astimezone()
    return dt.strftime("%Y-%m-%d %H:%M %Z")


def _render_tool_call(part: ResponsePart) -> str:
    lines = []
    label = part.tool_name or "tool"
    msg = part.tool_message.strip()
    lines.append(f"<details><summary>Tool: {label}</summary>")
    lines.append("")
    if msg:
        lines.append(msg)
    if part.text:
        lines.append("")
        lines.append("**Output:**")
        lines.append("")
        lines.append("```")
        lines.append(part.text.strip())
        lines.append("```")
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def _render_response(request: ChatRequest, include_tools: bool, include_thinking: bool) -> str:
    sections: list[str] = []

    for part in request.response_parts:
        if part.kind == "text" and part.text.strip():
            sections.append(part.text.strip())
        elif part.kind == "thinking" and include_thinking and part.text.strip():
            sections.append(
                "<details><summary>Thinking</summary>\n\n"
                + part.text.strip()
                + "\n\n</details>"
            )
        elif part.kind == "tool_call" and include_tools:
            sections.append(_render_tool_call(part))

    return "\n\n".join(sections)


def _render_request(req: ChatRequest, index: int, include_tools: bool, include_thinking: bool) -> str:
    lines: list[str] = []

    # User turn
    lines.append("---")
    lines.append("")
    ts = req.timestamp.astimezone().strftime("%H:%M:%S")
    lines.append(f"**User** _{ts}_")
    lines.append("")
    lines.append(req.user_text.strip())
    lines.append("")

    # Assistant turn
    response_md = _render_response(req, include_tools=include_tools, include_thinking=include_thinking)
    if response_md:
        lines.append(f"**Assistant**")
        lines.append("")
        lines.append(response_md)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render_session(
    session: ChatSession,
    include_tools: bool = True,
    include_thinking: bool = False,
) -> str:
    """Render a ChatSession as a Markdown document string."""
    lines: list[str] = []

    # Header
    lines.append(f"# {session.title}")
    lines.append("")
    lines.append(f"**Session ID:** `{session.session_id}`  ")
    lines.append(f"**Provider:** `{session.provider}`  ")
    lines.append(f"**Workspace:** `{session.workspace}`  ")
    lines.append(f"**Created:** {_fmt_ts(session)}  ")
    if session.model_id:
        lines.append(f"**Model:** `{session.model_id}`  ")
    lines.append("")

    user_reqs = session.user_requests
    if not user_reqs:
        lines.append("_No messages in this session._")
        return "\n".join(lines)

    for idx, req in enumerate(user_reqs):
        lines.append(_render_request(req, idx + 1, include_tools=include_tools, include_thinking=include_thinking))

    body = "\n".join(lines)
    # Collapse excessive blank lines
    body = _MULTI_NEWLINE_RE.sub("\n\n", body)
    return body.rstrip() + "\n"
