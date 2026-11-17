"""Tell real human prompts from the wrapper text agents record as user turns.

Claude Code stores slash commands, local command output, injected reminders,
continuation summaries and task notifications as user messages; Codex stores
its injected instructions and environment block the same way. None of those
are what the person typed, so titles and condensed views clean them first.
"""

from __future__ import annotations

import re

TITLE_LIMIT = 60

# Wrapper blocks agents prepend to a turn. Only a block at the very start of
# the message is a wrapper; the same tag mentioned mid-sentence is the
# person's own text and is kept.
_TAGS = (
    "system-reminder", "local-command-caveat", "local-command-stdout",
    "local-command-stderr", "task-notification", "command-name",
    "command-message", "command-args", "environment_context",
    "user_instructions", "INSTRUCTIONS", "ide_opened_file", "ide_selection",
    "user-prompt-submit-hook",
)
_ALT = "|".join(re.escape(t) for t in _TAGS)
_LEAD_BLOCK_RE = re.compile(r"\A\s*<(" + _ALT + r")(?:\s[^>]*)?>.*?</\1>", re.DOTALL)
# A wrapper tag at the start with no close (truncated record): all wrapper.
_LEAD_OPEN_RE = re.compile(r"\A\s*<(?:" + _ALT + r")(?:\s[^>]*)?>", re.DOTALL)
_COMMAND_NAME_RE = re.compile(r"<command-name>\s*(.*?)\s*</command-name>", re.DOTALL)
_COMMAND_ARGS_RE = re.compile(r"<command-args>\s*(.*?)\s*</command-args>", re.DOTALL)
_STDOUT_RE = re.compile(r"<local-command-std(?:out|err)>(.*?)</local-command-std(?:out|err)>", re.DOTALL)
_LEAD_COMMAND_RE = re.compile(r"\s*<command-(?:name|message)>")
_AGENTS_HEADER_RE = re.compile(r"^#\s*AGENTS\.md instructions for[^\n]*\n?", re.IGNORECASE)
_CONTINUATION = "This session is being continued from a previous conversation"


def is_continuation(text: str) -> bool:
    return text.lstrip().startswith(_CONTINUATION)


def slash_command(text: str) -> str:
    """The slash command a wrapper turn ran, with its arguments, or ''."""
    if not _LEAD_COMMAND_RE.match(text):
        return ""
    match = _COMMAND_NAME_RE.search(text)
    if not match:
        return ""
    name = match.group(1).strip()
    args = _COMMAND_ARGS_RE.search(text)
    arg_text = " ".join(args.group(1).split()) if args else ""
    return f"{name} {arg_text}".strip()


def human_text(text: str) -> str:
    """The prompt text left after removing wrapper blocks; '' if none."""
    if is_continuation(text):
        return ""
    cleaned = text
    while True:
        match = _LEAD_BLOCK_RE.match(cleaned)
        if match:
            cleaned = cleaned[match.end():]
            continue
        if _LEAD_OPEN_RE.match(cleaned):
            return ""
        break
    cleaned = _AGENTS_HEADER_RE.sub("", cleaned.strip())
    while True:
        match = _LEAD_BLOCK_RE.match(cleaned)
        if not match:
            break
        cleaned = cleaned[match.end():]
    return cleaned.strip()


def display_prompt(text: str) -> str:
    """A prompt as the condensed view shows it: wrappers become one short line."""
    stripped = text.strip()
    if is_continuation(stripped):
        return "[session continued from an earlier conversation]"
    command = slash_command(stripped)
    if command:
        return command
    human = human_text(stripped)
    if human:
        return human
    out = _STDOUT_RE.search(stripped) if stripped.startswith("<local-command-std") else None
    if out and out.group(1).strip():
        return "[local command output] " + " ".join(out.group(1).split())[:120]
    if _LEAD_BLOCK_RE.match(stripped) or _LEAD_OPEN_RE.match(stripped):
        return "[injected context]"
    return stripped


def short_title(text: str) -> str:
    """First line of the human prompt, trimmed; '' when there is no prompt."""
    human = human_text(text)
    for line in human.splitlines():
        line = line.lstrip("#").strip()
        if line:
            return line[:TITLE_LIMIT] + ("..." if len(line) > TITLE_LIMIT else "")
    return ""
