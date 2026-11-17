"""Search and retrieve parsed chat sessions for local integrations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal

from .parser import ChatSession
from .providers import PROVIDERS, DiscoveredSession, iter_discover
from .renderer import DEFAULT_MAX_ASSISTANT_CHARS, render_session

DEFAULT_LIMIT = 10
MAX_LIMIT = 50
DEFAULT_MAX_CHARS = 20_000
MAX_CHARS = 50_000
MAX_ASSISTANT_CHARS = 12_000
_TERM_RE = re.compile(r"\w+(?:[-']\w+)*", re.UNICODE)


@dataclass(frozen=True)
class SearchHit:
    provider: str
    workspace: str
    session_id: str
    title: str
    created_at: str
    matched_in: str
    snippet: str
    score: int


def _terms(query: str) -> list[str]:
    return [term.casefold() for term in _TERM_RE.findall(query) if term.strip()]


def _sections(session: ChatSession, include_tool_output: bool) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = [
        ("title", session.title),
        ("workspace", session.workspace),
    ]
    for index, request in enumerate(session.requests, start=1):
        if request.user_text:
            sections.append((f"turn {index} user", request.user_text))
        if request.assistant_text:
            sections.append((f"turn {index} assistant", request.assistant_text))
        if include_tool_output:
            for tool in request.tool_calls:
                detail = " ".join(
                    value for value in (tool.label, tool.summary, tool.output) if value
                )
                if detail:
                    sections.append((f"turn {index} tool", detail))
    return sections


def _snippet(text: str, terms: list[str], limit: int = 360) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    lowered = compact.casefold()
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    end = min(len(compact), start + limit)
    if end - start < limit:
        start = max(0, end - limit)
    prefix = "... " if start else ""
    suffix = " ..." if end < len(compact) else ""
    return prefix + compact[start:end].strip() + suffix


def _score(session: ChatSession, sections: list[tuple[str, str]], terms: list[str]) -> tuple[int, int]:
    title = session.title.casefold()
    score = 0
    best_index = 0
    best_matches = -1
    for index, (_, text) in enumerate(sections):
        lowered = text.casefold()
        matches = sum(lowered.count(term) for term in terms)
        if matches:
            score += matches
            if index < 2:
                score += matches * 4
            if matches > best_matches:
                best_index, best_matches = index, matches
    score += sum(title.count(term) * 8 for term in terms)
    return score, best_index


def _matching_context(
    sections: list[tuple[str, str]], terms: list[str]
) -> tuple[str, str]:
    matches = [
        (
            sum(text.casefold().count(term) for term in terms),
            {term for term in terms if term in text.casefold()},
            label,
            text,
        )
        for label, text in sections
        if any(term in text.casefold() for term in terms)
    ]
    remaining = set(terms)
    selected: list[tuple[int, set[str], str, str]] = []
    while matches and len(selected) < 2:
        matches.sort(key=lambda row: (len(row[1] & remaining), row[0]), reverse=True)
        chosen = matches.pop(0)
        selected.append(chosen)
        remaining -= chosen[1]
    return ", ".join(row[2] for row in selected), " | ".join(row[3] for row in selected)


def _validate_providers(providers: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(providers or PROVIDERS)
    unknown = sorted(set(selected) - set(PROVIDERS))
    if unknown:
        raise ValueError(f"Unknown provider(s): {', '.join(unknown)}")
    return selected


def _bounded(value: int, default: int, maximum: int) -> int:
    if value <= 0:
        return default
    return min(value, maximum)


def _recent_cutoff(recent_days: int | None) -> int | None:
    if recent_days is None:
        return None
    if recent_days <= 0:
        raise ValueError("recent_days must be greater than zero")
    cutoff = datetime.now(timezone.utc) - timedelta(days=recent_days)
    return int(cutoff.timestamp() * 1000)


def search_discovered(
    items: Iterable[DiscoveredSession],
    query: str = "",
    *,
    workspace: str | None = None,
    limit: int = DEFAULT_LIMIT,
    include_tool_output: bool = False,
    recent_days: int | None = None,
    sort_by: Literal["relevance", "recent"] = "relevance",
) -> dict[str, object]:
    """Search already discovered sessions without touching the source stores."""
    terms = _terms(query)
    if not terms and query.strip():
        raise ValueError("query must contain at least one searchable word")
    if sort_by not in ("relevance", "recent"):
        raise ValueError("sort_by must be 'relevance' or 'recent'")

    bounded_limit = _bounded(limit, DEFAULT_LIMIT, MAX_LIMIT)
    cutoff_ms = _recent_cutoff(recent_days)
    browse_recent = not terms
    results: list[tuple[int, int, SearchHit]] = []
    scanned = errors = matching = 0

    def result_key(row: tuple[int, int, SearchHit]) -> tuple:
        if browse_recent or sort_by == "recent":
            return (-row[1], row[2].title.casefold())
        return (-row[0], -row[1], row[2].title.casefold())

    def keep_result(score: int, session: ChatSession, hit: SearchHit) -> None:
        # Stable sorting preserves discovery order for exact ties.
        results.append((score, session.created_ms, hit))
        results.sort(key=result_key)
        del results[bounded_limit:]

    for item in items:
        if item.error:
            errors += 1
            continue
        session = item.session
        if session is None:
            continue
        if workspace and workspace.casefold() not in session.workspace.casefold():
            continue
        if cutoff_ms is not None and session.created_ms < cutoff_ms:
            continue
        scanned += 1
        if browse_recent:
            hit = SearchHit(
                provider=session.provider,
                workspace=session.workspace,
                session_id=session.session_id or item.path.stem,
                title=session.title,
                created_at=session.created_at.isoformat(),
                matched_in="",
                snippet="",
                score=0,
            )
            matching += 1
            keep_result(0, session, hit)
            continue
        sections = _sections(session, include_tool_output)
        remaining = set(terms)
        for _, section in sections:
            lowered = section.casefold()
            remaining.difference_update(term for term in terms if term in lowered)
            if not remaining:
                break
        if remaining:
            continue
        score, _ = _score(session, sections, terms)
        label, text = _matching_context(sections, terms)
        hit = SearchHit(
            provider=session.provider,
            workspace=session.workspace,
            session_id=session.session_id or item.path.stem,
            title=session.title,
            created_at=session.created_at.isoformat(),
            matched_in=label,
            snippet=_snippet(text, terms),
            score=score,
        )
        matching += 1
        keep_result(score, session, hit)
    return {
        "results": [asdict(row[2]) for row in results[:bounded_limit]],
        "searched_sessions": scanned,
        "matching_sessions": matching,
        "source_errors": errors,
        "search_mode": "recent" if browse_recent else sort_by,
        "recent_days": recent_days,
    }


def search_history(
    query: str = "",
    *,
    providers: Iterable[str] | None = None,
    workspace: str | None = None,
    limit: int = DEFAULT_LIMIT,
    include_tool_output: bool = False,
    recent_days: int | None = None,
    sort_by: Literal["relevance", "recent"] = "relevance",
) -> dict[str, object]:
    """Discover current local stores and search their parsed sessions."""
    selected = _validate_providers(providers)
    response = search_discovered(
        iter_discover(set(selected)),
        query,
        workspace=workspace,
        limit=limit,
        include_tool_output=include_tool_output,
        recent_days=recent_days,
        sort_by=sort_by,
    )
    response["providers"] = list(selected)
    return response


def _session_metadata(session: ChatSession) -> dict[str, object]:
    return {
        "provider": session.provider,
        "workspace": session.workspace,
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "turns": len(session.user_requests),
        "models": dict(session.models_used),
        "edited_files": session.edited_files,
    }


def read_session(
    provider: str,
    session_id: str,
    *,
    workspace: str | None = None,
    include_tools: bool = True,
    max_tool_output: int = 2_000,
    max_chars: int = DEFAULT_MAX_CHARS,
    condensed: bool = False,
    max_assistant_chars: int = DEFAULT_MAX_ASSISTANT_CHARS,
) -> dict[str, object]:
    """Find one session and render a bounded, readable transcript.

    Condensed mode is intended for cross-session review: it includes user
    prompts and assistant prose only, with no tool calls or operational
    metadata.
    """
    _validate_providers([provider])
    if not session_id.strip():
        raise ValueError("session_id must not be empty")
    bounded_chars = _bounded(max_chars, DEFAULT_MAX_CHARS, MAX_CHARS)
    for item in iter_discover({provider}):
        session = item.session
        if session is None:
            continue
        actual_id = session.session_id or item.path.stem
        if actual_id != session_id:
            continue
        if workspace and workspace.casefold() not in session.workspace.casefold():
            continue
        content = render_session(
            session,
            include_tools=include_tools,
            include_thinking=False,
            include_metrics=True,
            include_files=True,
            max_tool_output=max(0, max_tool_output),
            condensed=condensed,
            max_assistant_chars=min(max(0, max_assistant_chars), MAX_ASSISTANT_CHARS),
        )
        truncated = len(content) > bounded_chars
        if truncated:
            content = content[:bounded_chars].rstrip() + "\n\n... [transcript truncated]\n"
        return {
            "session": _session_metadata(session),
            "content": content,
            "truncated": truncated,
        }
    raise ValueError(f"Session not found: {provider}/{session_id}")
