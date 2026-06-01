"""
CLI entry point for copilot-exporter.

Usage:
    copilot-exporter [OPTIONS]

Options:
    --output-dir PATH        Directory to write Markdown files (default: ./export)
    --workspace NAME         Filter: only export sessions from workspaces whose
                             name contains NAME (case-insensitive). Repeatable.
    --session-id ID          Export only the session with this ID.
    --include-tools          Include tool call details in output (default: yes)
    --no-tools               Omit tool call details
    --include-thinking       Include model thinking/reasoning blocks
    --min-messages N         Skip sessions with fewer than N user messages (default: 1)
    --list                   List discovered workspaces and sessions, then exit
    --config-root PATH       Override VS Code User config directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .discovery import (
    find_workspace_storage_dirs,
    get_workspace_name,
    list_chat_session_files,
    _vscode_config_roots,
)
from .parser import parse_session
from .renderer import render_session, safe_filename


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="copilot-exporter",
        description="Export GitHub Copilot chat sessions to Markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output-dir", "-o",
        default="./export",
        metavar="PATH",
        help="Directory to write exported Markdown files (default: ./export)",
    )
    p.add_argument(
        "--workspace", "-w",
        action="append",
        default=[],
        metavar="NAME",
        dest="workspaces",
        help="Only export sessions from workspaces whose name contains NAME (case-insensitive, repeatable)",
    )
    p.add_argument(
        "--session-id",
        metavar="ID",
        help="Export only the session with this exact ID",
    )
    p.add_argument(
        "--no-tools",
        action="store_true",
        help="Omit tool call details from output",
    )
    p.add_argument(
        "--include-thinking",
        action="store_true",
        help="Include model thinking/reasoning blocks in output",
    )
    p.add_argument(
        "--min-messages",
        type=int,
        default=1,
        metavar="N",
        help="Skip sessions with fewer than N user messages (default: 1)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List discovered workspaces and sessions, then exit",
    )
    p.add_argument(
        "--config-root",
        metavar="PATH",
        help="Override the VS Code User config directory",
    )
    return p


def _workspace_matches(name: str, filters: list[str]) -> bool:
    if not filters:
        return True
    name_lower = name.lower()
    return any(f.lower() in name_lower for f in filters)


# ---------------------------------------------------------------------------
# List mode
# ---------------------------------------------------------------------------

def _run_list(ws_dirs: list[Path], workspace_filters: list[str]) -> None:
    found_any = False
    for ws_dir in ws_dirs:
        ws_name = get_workspace_name(ws_dir)
        if not _workspace_matches(ws_name, workspace_filters):
            continue
        session_files = list_chat_session_files(ws_dir)
        if not session_files:
            continue
        found_any = True
        print(f"\n{ws_name}")
        print(f"  Storage: {ws_dir}")
        for sf in session_files:
            session = parse_session(sf)
            if session:
                n = len(session.user_requests)
                print(f"  [{session.session_id}] {session.title!r} ({n} messages)")
            else:
                print(f"  [{sf.stem}] (empty/unreadable)")
    if not found_any:
        print("No workspaces with chat sessions found.")


# ---------------------------------------------------------------------------
# Export mode
# ---------------------------------------------------------------------------

def _run_export(
    ws_dirs: list[Path],
    workspace_filters: list[str],
    session_id_filter: str | None,
    output_root: Path,
    include_tools: bool,
    include_thinking: bool,
    min_messages: int,
) -> None:
    exported = 0
    skipped = 0

    for ws_dir in ws_dirs:
        ws_name = get_workspace_name(ws_dir)
        if not _workspace_matches(ws_name, workspace_filters):
            continue

        session_files = list_chat_session_files(ws_dir)
        for sf in session_files:
            if session_id_filter and sf.stem != session_id_filter:
                continue

            session = parse_session(sf)
            if session is None:
                skipped += 1
                continue
            if len(session.user_requests) < min_messages:
                skipped += 1
                continue

            # Derive output path: <output_root>/<WorkspaceName>/<SessionTitle>.md
            ws_dir_name = safe_filename(ws_name)
            session_file_name = safe_filename(session.title) + ".md"
            out_path = output_root / ws_dir_name / session_file_name

            # Avoid overwriting different sessions with the same sanitized title
            if out_path.exists():
                out_path = output_root / ws_dir_name / (
                    safe_filename(session.title) + f"_{session.session_id[:8]}.md"
                )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            md = render_session(
                session,
                include_tools=include_tools,
                include_thinking=include_thinking,
            )
            out_path.write_text(md, encoding="utf-8")
            print(f"  Exported: {out_path.relative_to(output_root)}")
            exported += 1

    print(f"\nDone. {exported} session(s) exported, {skipped} skipped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config_root = Path(args.config_root) if args.config_root else None
    ws_dirs = find_workspace_storage_dirs(config_root)

    if not ws_dirs:
        roots = _vscode_config_roots() if config_root is None else [config_root]
        print(
            "ERROR: No VS Code workspace storage directories found.\n"
            "Searched:\n" + "\n".join(f"  {r}" for r in roots),
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list:
        _run_list(ws_dirs, args.workspaces)
        return

    output_root = Path(args.output_dir)
    _run_export(
        ws_dirs=ws_dirs,
        workspace_filters=args.workspaces,
        session_id_filter=args.session_id,
        output_root=output_root,
        include_tools=not args.no_tools,
        include_thinking=args.include_thinking,
        min_messages=args.min_messages,
    )


if __name__ == "__main__":
    main()
