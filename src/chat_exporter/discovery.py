"""
Discover VS Code workspace storage directories and identify workspaces.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


# ---------------------------------------------------------------------------
# Config root detection
# ---------------------------------------------------------------------------

def _vscode_config_roots() -> list[Path]:
    """Return candidate VS Code User config directories for the current OS."""
    system = platform.system()
    home = Path.home()

    if system == "Linux":
        xdg = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
        roots = [
            xdg / "Code" / "User",
            xdg / "Code - Insiders" / "User",
            xdg / "Cursor" / "User",
            xdg / "VSCodium" / "User",
        ]
    elif system == "Darwin":
        base = home / "Library" / "Application Support"
        roots = [
            base / "Code" / "User",
            base / "Code - Insiders" / "User",
            base / "Cursor" / "User",
            base / "VSCodium" / "User",
        ]
    elif system == "Windows":
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        roots = [
            appdata / "Code" / "User",
            appdata / "Code - Insiders" / "User",
            appdata / "Cursor" / "User",
            appdata / "VSCodium" / "User",
        ]
    else:
        roots = []

    return [r for r in roots if r.is_dir()]


def find_workspace_storage_dirs(config_root: Path | None = None) -> list[Path]:
    """Return all hashed workspace dirs under workspaceStorage."""
    if config_root is not None:
        roots = [config_root]
    else:
        roots = _vscode_config_roots()

    dirs: list[Path] = []
    for root in roots:
        ws_storage = root / "workspaceStorage"
        if not ws_storage.is_dir():
            continue
        for entry in sorted(ws_storage.iterdir()):
            if entry.is_dir() and (entry / "workspace.json").exists():
                dirs.append(entry)
    return dirs


# ---------------------------------------------------------------------------
# Workspace identification
# ---------------------------------------------------------------------------

def _uri_to_display_name(uri: str) -> str:
    """Convert a file URI or path string to a human-readable project name."""
    try:
        parsed = urlparse(uri)
        if parsed.scheme in ("file", "vscode-remote"):
            path = unquote(parsed.path)
        else:
            path = unquote(uri)
        # Use last non-empty path component
        parts = [p for p in path.split("/") if p]
        return parts[-1] if parts else uri
    except Exception:
        return uri


def get_workspace_name(ws_dir: Path) -> str:
    """Read workspace.json and return a display name for the workspace."""
    ws_json = ws_dir / "workspace.json"
    try:
        data = json.loads(ws_json.read_text(encoding="utf-8"))
    except Exception:
        return ws_dir.name

    # Single-folder workspace
    folder = data.get("folder")
    if folder:
        return _uri_to_display_name(folder)

    # Multi-root workspace file
    workspace_file = data.get("workspace")
    if workspace_file:
        return _uri_to_display_name(workspace_file)

    return ws_dir.name


def list_chat_session_files(ws_dir: Path) -> list[Path]:
    """Return all .jsonl chat session files in the workspace directory."""
    chat_dir = ws_dir / "chatSessions"
    if not chat_dir.is_dir():
        return []
    return sorted(chat_dir.glob("*.jsonl"))
