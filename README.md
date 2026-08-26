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
  --format {md,json}       Output format (default: md); json emits the full
                           structured record
  --no-tools               Omit tool calls and results
  --max-tool-output N      Truncate each tool output to N characters
                           (default: 2000; 0 = unlimited)
  --tool-args              Include the JSON arguments each tool was called with
  --include-thinking       Include stored reasoning blocks
  --no-metrics             Omit per-turn model attribution, timing and tokens
  --no-file-edits          Omit the files-changed summaries
  --include-context        List the context files attached to each turn
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

# Full detail: reasoning, tool arguments, untruncated output
chat-exporter --include-thinking --tool-args --max-tool-output 0

# Structured JSON for further processing
chat-exporter --format json -o ~/chat-json

# Reading copy: prose only, no tool noise or metrics
chat-exporter --no-tools --no-metrics --no-file-edits
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

Each Markdown file opens with a session header: provider, workspace, session
ID, creation date, turn count, wall-clock span, every model used with a
per-model turn count, a tool-call histogram, total generation time, token
totals, the git branch, and the client version.

Every turn is a User/Assistant pair whose assistant line carries its own
attribution: the model that answered that turn, how long it took, time to
first token, and its token counts - output, and where the provider records
them, input, cached, and reasoning tokens. Models are recorded per request, so
a session where you switched models mid-conversation reports each turn
accurately instead of assuming one model throughout. A turn that ended for an
unusual reason, such as hitting a length limit, says so.

Tool invocations render as collapsed `<details>` blocks unless `--no-tools`,
carrying the command, working directory, exit code and duration for terminal
calls, captured output truncated at `--max-tool-output`, and argument JSON
with `--tool-args`. Todo lists render as checklists, and subagent calls show
their own model, prompt and result.

Turn outcomes such as cancellations, rate limits, network errors, and
length-limit truncations are called out rather than silently producing a
short answer. Files changed are summarized per turn and per session unless
`--no-file-edits`; `--include-context` additionally lists the files attached
to each turn.

Depth varies by provider, since each stores a different amount. Copilot,
Claude Code, Codex, and OpenCode all record tool output, per-turn timing, and
token usage; shell exit codes come from Copilot, Codex, and OpenCode, and
Claude Code contributes its subagent transcripts, joined back onto the call
that launched them. AGY stores the least, since its transcripts are decoded
from an undocumented protobuf. Fields a format does not preserve are simply
omitted.

`--format json` writes the complete parsed record for feeding into other
tooling: every field above, plus tool call IDs, canonical tool names,
per-call error flags, and raw timing values.

The exporter opens source stores read-only. It writes only beneath the selected
output directory.

## Copilot editor support

Copilot workspace discovery supports VS Code, VS Code Insiders, Cursor, and
VSCodium on Linux, macOS, and Windows. Use `--config-root` for a non-standard
VS Code installation.
