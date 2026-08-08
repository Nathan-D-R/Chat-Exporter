"""CLI entry point for multi-provider chat export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .providers import PROVIDERS, DiscoveredSession, discover
from .renderer import render_session, safe_filename


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat-exporter",
        description="Export local AI coding chat sessions to Markdown files.",
    )
    parser.add_argument("--output-dir", "-o", default="./export", metavar="PATH")
    parser.add_argument("--provider", action="append", choices=PROVIDERS, default=[],
                        help="Provider to export (repeatable; default: all)")
    parser.add_argument("--workspace", "-w", action="append", default=[], dest="workspaces")
    parser.add_argument("--session-id", metavar="ID")
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument("--include-thinking", action="store_true")
    parser.add_argument("--min-messages", type=int, default=1, metavar="N")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--config-root", metavar="PATH",
                        help="Override the VS Code User config directory for Copilot")
    return parser


def _workspace_matches(name: str, filters: list[str]) -> bool:
    return not filters or any(value.lower() in name.lower() for value in filters)


def _filtered(items: list[DiscoveredSession], args: argparse.Namespace) -> list[DiscoveredSession]:
    result = []
    for item in items:
        if not _workspace_matches(item.workspace, args.workspaces):
            continue
        if args.session_id and item.path.stem != args.session_id and (
            not item.session or item.session.session_id != args.session_id
        ):
            continue
        result.append(item)
    return result


def _run_list(items: list[DiscoveredSession]) -> None:
    if not items:
        print("No chat sessions found for the selected providers.")
        return
    previous: tuple[str, str] | None = None
    for item in items:
        group = item.provider, item.workspace
        if group != previous:
            print(f"\n{item.provider} / {item.workspace}")
            previous = group
        if item.session:
            print(f"  [{item.session.session_id}] {item.session.title!r} "
                  f"({len(item.session.user_requests)} messages)")
        else:
            detail = f": {item.error}" if item.error else ""
            print(f"  [{item.path.stem}] (empty/unreadable{detail})")


def _run_export(items: list[DiscoveredSession], args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    exported = skipped = 0
    for item in items:
        session = item.session
        if session is None or len(session.user_requests) < args.min_messages:
            skipped += 1
            continue
        directory = output_root / safe_filename(session.provider) / safe_filename(session.workspace)
        out_path = directory / (safe_filename(session.title) + ".md")
        if out_path.exists():
            out_path = directory / (safe_filename(session.title) + f"_{session.session_id[:8]}.md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_session(session, not args.no_tools, args.include_thinking), encoding="utf-8")
        print(f"  Exported: {out_path.relative_to(output_root)}")
        exported += 1
    print(f"\nDone. {exported} session(s) exported, {skipped} skipped.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.min_messages < 0:
        parser.error("--min-messages must be zero or greater")
    providers = set(args.provider or PROVIDERS)
    items = _filtered(discover(providers, Path(args.config_root) if args.config_root else None), args)
    try:
        if args.list:
            _run_list(items)
        elif not items:
            print("ERROR: No chat sessions found for the selected providers.", file=sys.stderr)
            sys.exit(1)
        else:
            _run_export(items, args)
    except BrokenPipeError:
        sys.stdout.close()


if __name__ == "__main__":
    main()
