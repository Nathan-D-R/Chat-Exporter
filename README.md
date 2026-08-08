# Chat Exporter

Export local AI coding chat sessions to structured Markdown files, organized by
provider and workspace.

The Python distribution and command are named `chat-exporter`.

## Supported providers

| Provider | Local store |
|---|---|
| GitHub Copilot | VS Code `User/workspaceStorage/*/chatSessions` |
| AGY / Antigravity CLI | `~/.gemini/antigravity-cli` |
| Claude Code | `~/.claude/projects` |
| Codex | `$CODEX_HOME/sessions` or `~/.codex/sessions` |
| OpenCode | `$XDG_DATA_HOME/opencode` or `~/.local/share/opencode` |

AGY is decoded directly from its read-only SQLite/protobuf store. OpenCode
support handles both its legacy JSON layout and current `opencode*.db` store. Provider formats are not
stable public interchange formats, so test an export after upgrading a client.

## Installation

With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install .
```

Or run without installing:

```bash
uv run chat-exporter [OPTIONS]
```

Plain Python 3.11 or newer also works:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```text
chat-exporter [OPTIONS]

Options:
  -o, --output-dir PATH    Output directory (default: ./export)
  --provider NAME          Export one provider; repeatable (default: all)
  -w, --workspace NAME     Filter workspace name; repeatable
  --session-id ID          Export one exact session ID
  --no-tools               Omit tool calls and results
  --include-thinking       Include stored reasoning blocks
  --min-messages N         Minimum user messages (default: 1)
  --list                   List discovered sessions without exporting
  --config-root PATH       Override VS Code User config root for Copilot
```

Provider names are `copilot`, `agy`, `claude`, `codex`, and `opencode`.

## Examples

```bash
# List every locally discoverable session
chat-exporter --list

# Export Claude Code and Codex only
chat-exporter --provider claude --provider codex

# Export one workspace without tool output
chat-exporter -w MyProject --no-tools -o ~/Documents/chat-export
```

## Output

```text
export/
  claude/
    MyProject/
      Session title.md
  codex/
    MyProject/
      Another session.md
```

Each Markdown file includes provider, workspace, session ID, creation date,
model when available, user and assistant turns, optional tool details, and
optional stored thinking blocks.

The exporter opens source stores read-only. It writes only beneath the selected
output directory.

## Copilot editor support

Copilot workspace discovery supports VS Code, VS Code Insiders, Cursor, and
VSCodium on Linux, macOS, and Windows. Use `--config-root` for a non-standard
VS Code installation.
