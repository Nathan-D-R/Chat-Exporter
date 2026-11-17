"""Search and retrieve parsed chat sessions for local integrations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal

from .parser import ChatSession
from .providers import PROVIDERS, DiscoveredSession, iter_discover
from .renderer import (
    DEFAULT_MAX_ASSISTANT_CHARS,
    _collapse_blank_lines,
    render_session_parts,
    resolve_detail,
)

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
    last_activity_at: str | None = None
    turns: int = 0


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


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
                last_activity_at=_iso(session.last_activity_at),
                turns=session.turn_count,
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
            last_activity_at=_iso(session.last_activity_at),
            turns=session.turn_count,
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
        "turns": session.turn_count,
        "last_activity_at": _iso(session.last_activity_at),
        "models": dict(session.models_used),
        "edited_files": session.edited_files,
    }


_NOTICE_RESERVE = 240


def _page_blocks(blocks: list[str], budget: int, from_end: bool) -> list[tuple[int, int]]:
    """Split block indexes into pages of at most `budget` characters, whole
    blocks only (a block longer than the budget gets a page to itself). Pages
    are returned in reading order: from the end when `from_end`, else from
    the start. Each page is a (first, stop) index range."""
    order = range(len(blocks) - 1, -1, -1) if from_end else range(len(blocks))
    pages: list[tuple[int, int]] = []
    used = 0
    first = None
    last = None
    for i in order:
        size = len(blocks[i]) + 1
        if first is not None and used + size > budget:
            pages.append((min(first, last), max(first, last) + 1))
            first, used = None, 0
        if first is None:
            first = i
        last = i
        used += size
    if first is not None:
        pages.append((min(first, last), max(first, last) + 1))
    return pages


def _split_block(text: str, size: int) -> list[str]:
    """Cut one oversized turn into chunks of at most `size` characters at line
    boundaries (a longer line is cut), with a marker on each side that continues."""
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for line in text.split("\n"):
        while len(line) > size:
            if current:
                pieces.append("\n".join(current))
                current, used = [], 0
            pieces.append(line[:size])
            line = line[size:]
        if current and used + len(line) + 1 > size:
            pieces.append("\n".join(current))
            current, used = [], 0
        current.append(line)
        used += len(line) + 1
    if current:
        pieces.append("\n".join(current))
    last = len(pieces) - 1
    return [
        ("[start of turn truncated; see the previous part]\n\n" if i else "")
        + piece
        + ("\n\n[turn continues in the next part]" if i < last else "")
        for i, piece in enumerate(pieces)
    ]


def _page_notice(before: int, after: int, page: int, from_end: bool) -> tuple[str, str]:
    """Markers for the turns left out before and after a page."""
    previous = page + 1 if from_end else page - 1
    following = page - 1 if from_end else page + 1
    top = bottom = ""
    if before:
        top = f"[{before} earlier turns omitted; page={previous} for the previous part]"
    if after:
        bottom = f"[{after} later turns omitted; page={following} for the next part]"
    return top, bottom


def read_session(
    provider: str,
    session_id: str,
    *,
    workspace: str | None = None,
    include_tools: bool = True,
    max_tool_output: int = 2_000,
    max_chars: int = DEFAULT_MAX_CHARS,
    condensed: bool | None = None,
    max_assistant_chars: int = DEFAULT_MAX_ASSISTANT_CHARS,
    detail: str | None = None,
    from_end: bool = True,
    page: int = 0,
) -> dict[str, object]:
    """Find one session and render a bounded, readable transcript.

    detail is "condensed" (default), "standard" or "full". The legacy
    `condensed` flag applies only when detail is not given: True means
    "condensed", False means "full". When the transcript is longer than
    max_chars it is cut between turns: from_end=True keeps the last turns
    (page 0), and page=1, 2, ... read the parts before it; from_end=False
    keeps the first turns and pages forward. A single turn longer than a page
    is split at line boundaries across pages, each part marked, so page= reaches
    all of it (from_end=True shows the end of that turn on page 0). The result
    reports truncated, page and pages. Turn numbers mean the same at every
    detail level.
    """
    _validate_providers([provider])
    if not session_id.strip():
        raise ValueError("session_id must not be empty")
    if page < 0:
        raise ValueError("page must be zero or greater")
    level = resolve_detail(detail, condensed)
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
        header, blocks, footer = render_session_parts(
            session,
            include_tools=include_tools,
            include_thinking=False,
            include_metrics=True,
            include_files=True,
            max_tool_output=max(0, max_tool_output),
            max_assistant_chars=min(max(0, max_assistant_chars), MAX_ASSISTANT_CHARS),
            detail=level,
            group_notifications=True,
        )
        if footer and blocks:
            blocks[-1] = blocks[-1] + "\n" + footer
        elif footer:
            header = header + "\n" + footer
        budget = max(bounded_chars - len(header) - _NOTICE_RESERVE, 200)
        # A turn longer than a page is split so every part is reachable.
        owner: list[int] = []
        pieces: list[str] = []
        for turn, block in enumerate(blocks):
            chunks = _split_block(block, max(budget - 100, 100)) if len(block) + 1 > budget else [block]
            pieces += chunks
            owner += [turn] * len(chunks)
        blocks = pieces
        pages = _page_blocks(blocks, budget, from_end)
        total = max(len(pages), 1)
        if page >= total:
            raise ValueError(f"page {page} is out of range; this transcript has {total} page(s)")
        if pages:
            first, stop = pages[page]
            chosen = blocks[first:stop]
            shown = set(owner[first:stop])
            before = len(set(owner[:first]) - shown)
            after = len(set(owner[stop:]) - shown)
            top, bottom = _page_notice(before, after, page, from_end)
            parts = [header] + ([top] if top else []) + chosen + ([bottom] if bottom else [])
        else:
            parts = [header]
        content = _collapse_blank_lines("\n".join(parts)).rstrip() + "\n"
        return {
            "session": _session_metadata(session),
            "content": content,
            "truncated": total > 1,
            "page": page,
            "pages": total,
            "detail": level,
            "from_end": from_end,
        }
    raise ValueError(f"Session not found: {provider}/{session_id}")
