# copilot-exporter

Export GitHub Copilot chat sessions from VS Code to structured Markdown files,
organized by workspace.

## Installation

### With uv (recommended)

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the tool globally:

```bash
uv tool install .
```

Or run directly without installing (uv manages the virtualenv automatically):

```bash
uv run copilot-exporter [OPTIONS]
```

### With plain Python

```bash
git clone https://github.com/Nathan-D-R/Copilot-Exporter
cd copilot-exporter
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
copilot-exporter [OPTIONS]
```

## Usage

```
copilot-exporter [OPTIONS]

Options:
  -o, --output-dir PATH    Directory to write Markdown files (default: ./export)
  -w, --workspace NAME     Only export sessions from workspaces whose name
                           contains NAME (case-insensitive, repeatable)
  --session-id ID          Export only the session with this exact ID
  --no-tools               Omit tool call details from output
  --include-thinking       Include model thinking/reasoning blocks
  --min-messages N         Skip sessions with fewer than N user messages
                           (default: 1)
  --list                   List discovered workspaces and sessions, then exit
  --config-root PATH       Override the VS Code User config directory
```

## Examples

```bash
# List all workspaces and sessions
copilot-exporter --list

# Export everything to ./export/
copilot-exporter

# Export only sessions from a specific workspace
copilot-exporter -w MyProject -o ~/Documents/copilot-export

# Export sessions with at least 5 messages, without tool call details
copilot-exporter --min-messages 5 --no-tools

# Export a single session by ID
copilot-exporter --session-id 93f92856-c448-4328-ab10-a13c3a059e1e
```

## Output structure

```
export/
  WorkspaceName/
    Session Title.md
    Another Session.md
  OtherProject/
    ...
```

Each Markdown file contains:

- Session metadata (ID, creation date, model)
- Each turn as a User/Assistant pair with timestamps
- Tool invocations as collapsed `<details>` blocks (unless `--no-tools`)
- Thinking blocks optionally included with `--include-thinking`

## Supported editors

- VS Code
- VS Code Insiders
- Cursor
- VSCodium

## Cross-platform support

Config directories are detected automatically:

| OS      | Path                                                       |
|---------|------------------------------------------------------------|
| Linux   | `~/.config/Code/User/workspaceStorage`                     |
| macOS   | `~/Library/Application Support/Code/User/workspaceStorage` |
| Windows | `%APPDATA%\Code\User\workspaceStorage`                     |

Use `--config-root` to override for non-standard installation paths.
