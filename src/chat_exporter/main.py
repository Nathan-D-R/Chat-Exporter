"""CLI entry point for multi-provider chat export."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .providers import PROVIDERS, DiscoveredSession, discover
from .renderer import (
    DEFAULT_MAX_TOOL_OUTPUT,
    render_session,
    render_session_json,
    safe_filename,
)


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
    parser.add_argument("--no-tools", action="store_true",
                        help="Omit tool call details from output")
    parser.add_argument("--max-tool-output", type=int, default=DEFAULT_MAX_TOOL_OUTPUT, metavar="N",
                        help=f"Truncate each tool output to N characters, 0 for unlimited "
                             f"(default: {DEFAULT_MAX_TOOL_OUTPUT})")
    parser.add_argument("--tool-args", action="store_true",
                        help="Include the JSON arguments each tool was called with")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Omit per-turn model attribution, timing and token counts")
    parser.add_argument("--no-file-edits", action="store_true",
                        help="Omit the per-turn and per-session files-changed summaries")
    parser.add_argument("--include-context", action="store_true",
                        help="List the context files attached to each turn")
    parser.add_argument("--format", choices=("md", "json"), default="md",
                        help="Output format (default: md). json emits the full structured record.")
    parser.add_argument("--include-thinking", action="store_true",
                        help="Include model thinking/reasoning blocks in output")
    parser.add_argument("--min-messages", type=int, default=1, metavar="N")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--config-root", metavar="PATH",
                        help="Override the VS Code User config directory for Copilot"
                             " (alias for --root copilot=PATH)")
    parser.add_argument("--root", action="append", default=[], metavar="NAME=PATH",
                        dest="roots",
                        help="Read a provider's sessions from PATH instead of its default"
                             f" location; repeatable. NAME is one of: {', '.join(PROVIDERS)}")
    return parser


@dataclass
class RenderOptions:
    """Everything that controls what ends up in an exported file."""
    fmt: str = "md"
    include_tools: bool = True
    include_thinking: bool = False
    include_metrics: bool = True
    include_files: bool = True
    include_context: bool = False
    include_tool_args: bool = False
    max_tool_output: int = DEFAULT_MAX_TOOL_OUTPUT

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RenderOptions":
        return cls(
            fmt=args.format,
            include_tools=not args.no_tools,
            include_thinking=args.include_thinking,
            include_metrics=not args.no_metrics,
            include_files=not args.no_file_edits,
            include_context=args.include_context,
            include_tool_args=args.tool_args,
            max_tool_output=max(0, args.max_tool_output),
        )

    @property
    def suffix(self) -> str:
        return ".json" if self.fmt == "json" else ".md"

    def render(self, session) -> str:
        if self.fmt == "json":
            return render_session_json(
                session,
                include_thinking=self.include_thinking,
                max_tool_output=self.max_tool_output,
            )
        return render_session(
            session,
            include_tools=self.include_tools,
            include_thinking=self.include_thinking,
            include_metrics=self.include_metrics,
            include_files=self.include_files,
            include_context=self.include_context,
            include_tool_args=self.include_tool_args,
            max_tool_output=self.max_tool_output,
        )


def _parse_roots(parser: argparse.ArgumentParser, values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        name, sep, raw = value.partition("=")
        name, raw = name.strip(), raw.strip()
        if not sep or not name or not raw:
            parser.error(f"--root expects NAME=PATH, got {value!r}")
        if name not in PROVIDERS:
            parser.error(f"--root: unknown provider {name!r}; choose from {', '.join(PROVIDERS)}")
        path = Path(raw).expanduser()
        if not path.is_dir():
            parser.error(f"--root: {path} is not a directory")
        roots[name] = path
    return roots


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
            detail = [f"{len(item.session.user_requests)} messages"]
            models = item.session.models_used
            if models:
                detail.append("/".join(name for name, _ in models[:2]))
            if item.session.total_tool_calls:
                detail.append(f"{item.session.total_tool_calls} tool calls")
            print(f"  [{item.session.session_id}] {item.session.title!r} "
                  f"({', '.join(detail)})")
        else:
            detail = f": {item.error}" if item.error else ""
            print(f"  [{item.path.stem}] (empty/unreadable{detail})")


def _run_export(items: list[DiscoveredSession], args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    options = RenderOptions.from_args(args)
    exported = skipped = 0
    for item in items:
        session = item.session
        if session is None or len(session.user_requests) < args.min_messages:
            skipped += 1
            continue
        directory = output_root / safe_filename(session.provider) / safe_filename(session.workspace)
        stem = safe_filename(session.title)
        out_path = directory / (stem + options.suffix)
        if out_path.exists():
            out_path = directory / f"{stem}_{session.session_id[:8]}{options.suffix}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(options.render(session), encoding="utf-8")
        print(f"  Exported: {out_path.relative_to(output_root)}")
        exported += 1
    print(f"\nDone. {exported} session(s) exported, {skipped} skipped.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.min_messages < 0:
        parser.error("--min-messages must be zero or greater")
    providers = set(args.provider or PROVIDERS)
    roots = _parse_roots(parser, args.roots)
    config_root = Path(args.config_root) if args.config_root else None
    items = _filtered(discover(providers, config_root, roots), args)
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
