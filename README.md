# Chat Exporter

Export local AI coding chat sessions to structured Markdown files, organized by
provider and workspace, to Parquet tables for analysis, and to a read-only MCP
server that searches past chats.

The Python distribution and command are named `chat-exporter`.

## Supported providers

| Provider | Name | Local store |
|---|---|---|
| GitHub Copilot, VS Code extension | `copilot` | VS Code `User/workspaceStorage/*/chatSessions` |
| GitHub Copilot CLI | `copilot-cli` | `~/.copilot/session-state` |
| GitHub Copilot Desktop | `copilot-desktop` | `~/.copilot/session-state`, titles from `~/.copilot/data.db` |
| AGY / Antigravity CLI | `agy` | `~/.gemini/antigravity-cli` |
| Claude Code | `claude` | `~/.claude/projects` |
| Codex | `codex` | `$CODEX_HOME/sessions` or `~/.codex/sessions` |
| OpenCode | `opencode` | `$XDG_DATA_HOME/opencode` or `~/.local/share/opencode` |

The three Copilot entries are separate products with separate stores: the VS Code
extension, the standalone CLI and the desktop app. Exporting one does not cover
the others, and all are included by default.

For the CLI, the event log carries the conversation and its tool calls while a
sqlite index alongside it carries per-turn tokens and timing; builds older than
usage tracking still export, without those figures. AGY is decoded directly from
its read-only SQLite/protobuf store. OpenCode support handles both its legacy
JSON layout and current `opencode*.db` store. Provider formats are not stable
public interchange formats, so test an export after upgrading a client.

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

Parquet output needs polars, an optional extra: `uv sync --extra parquet`
(or `pip install -e ".[parquet]"`). Markdown and JSON export and the MCP
server do not need it.

## Usage

```text
chat-exporter [OPTIONS]

Options:
  -o, --output-dir PATH    Output directory (default: ./export)
  --provider NAME          Export one provider; repeatable (default: all)
  -w, --workspace NAME     Filter workspace name; repeatable
  --session-id ID          Export one exact session ID
  --format {md,json,parquet}
                           Output format (default: md); json emits the full
                           structured record; parquet writes five tables
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
  --root NAME=PATH         Read a provider's sessions from PATH instead of its
                           default location; repeatable
  --config-root PATH       Override VS Code User config root for Copilot
                           (alias for --root copilot=PATH)
```

Provider names are `copilot`, `copilot-cli`, `copilot-desktop`, `agy`, `claude`, `codex`, and `opencode`.

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

# Tables for analysis: sessions, turns, tool_calls, events and permissions (.parquet)
chat-exporter --format parquet --provider copilot-cli -o ~/chat-tables

# Reading copy: prose only, no tool noise or metrics
chat-exporter --no-tools --no-metrics --no-file-edits

# Export a store from somewhere other than the current home: one copied out
# of a container, restored from a backup, or belonging to another user
chat-exporter --root copilot-cli=<BACKUP_ROOT>/.copilot
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

## Parquet output

`--format parquet` writes five files directly into the output directory,
covering every exported session; there are no provider folders or per-session
files:

| File | One row is |
|---|---|
| `sessions.parquet` | a session: identity, workspace, models used, turn and tool-call counts, first and last turn times, token sums |
| `turns.parquet` | a user request and its response: text, model, mode, timing, token counts, errors, context and edited files |
| `tool_calls.parquet` | a tool call within a turn, ordered by `turn_index` and `call_index`; `part_index` places it among the turn's response parts. For GitHub Copilot CLI and desktop sessions, a failed call also fills `error_code`, `error_message`, `failure_category` and `failure_stage`, and every completed call records `context_tokens` and `context_limit` (the context size when it ran) |
| `events.parquet` | a line of a session's `events.jsonl` (GitHub Copilot CLI and desktop only): event type and time, ids, the sorted keys of `data`, and `data` as JSON cut to 4,000 characters (`data_chars` keeps the full length). `scalars_json` is never cut: the top-level `data` entries that are a bool, a number or a string of at most 300 characters. `nested_json` holds the scalar leaves of `data.result`, `data.error` and `data.assistedApprovalAssessment` |
| `permissions.parquet` | a `permission.requested` event joined to its `permission.completed` event by `requestId`: kind, command, path or URL, tool, the decision (`result_kind`, `decision_source`) and `wait_ms`, plus the request's `agent_mode`, `permission_mode`, `recovery_episode_id` and `assisted_assessment_json` (cut to 1,000 characters), `approved` and `outcome` (`approved`, `approved_saved`, `denied_by_user`, `denied_unattended`, `cancelled`, `pending` or `other`); the completion columns are empty and `outcome` is `pending` for a request that was never decided |

Tables join on `session_id` and, for tool calls, `turn_index`; events and
permissions also carry `tool_call_id`, which matches `tool_calls.call_id`. Timestamps are
UTC. Token and timing columns are null where the provider does not record them.
`--provider`, `-w`, `--session-id`, `--min-messages` and `--root` filter as
usual. `--max-tool-output` truncates `tool_calls.output` (`output_chars` keeps
the original length). `--include-thinking` fills `turns.thinking_text`.
`arguments` is always written, regardless of `--tool-args`. `--no-tools` leaves
`tool_calls.parquet` empty but keeps the tool counts in the other tables. A blank
or unparsable `events.jsonl` line is skipped and counted in a log warning.
`--no-metrics`, `--no-file-edits` and `--include-context` do not apply. Read the
files with any Parquet reader, for example `polars.read_parquet`.

## Chat-history MCP server

`chat-exporter-mcp` runs a read-only `chat-history` MCP server over stdio. Register it
with any MCP client, for example:

```bash
uv run --directory <CHECKOUT> --frozen chat-exporter-mcp
# or, equivalently, from the checkout root:
uv run python mcp_server.py
```

It reads the current local provider stores directly, so an export is not required
before searching. The `search_chat_history` tool returns ranked snippets and
session IDs; pass a result to `read_chat_session` for a bounded transcript.
`read_chat_session` takes a `detail` level:

- `condensed` (default): user prompts, assistant prose, and one line per
  subagent or background-task result (status plus the first 200 characters of
  the result). No tool calls, tool output, or metrics.
- `standard`: condensed plus one line per tool call with its name, a short
  argument preview, and ok or error.
- `full`: tool calls with output (bounded by `max_tool_output`), metrics, and
  file lists.

The older `condensed` flag still works when `detail` is not given: true means
`condensed`, false means `full`. In condensed and standard output, an assistant
paragraph (40 characters or more) that already appeared earlier in the session
shows as `[repeated from turn N]`. Each turn's assistant prose is bounded by
`max_assistant_chars` (2,000 by default).

A transcript longer than `max_chars` (20,000 by default, 50,000 at most) is
cut between turns. By default (`from_end=true`) the session header and the
last turns are kept, which is the part needed to pick up where a session
stopped; `page=1` returns the part before that, `page=2` the part before it,
and so on. With `from_end=false` the first turns are kept and `page` moves
forward. A single turn longer than a page is split at line boundaries across
pages, each part marked, so `page` reaches all of it. Turn numbers are the same
at every detail level (a background-task notification joins the turn before it). The result reports `truncated`, `page` and `pages`. Each
`search_chat_history` result carries `last_activity_at` and `turns`.

The server searches all supported providers by default. Restrict a search with
provider names (`copilot`, `copilot-cli`, `copilot-desktop`, `claude`, `codex`,
`agy`, or `opencode`) or a workspace substring. Use `recent_days` to restrict results to
sessions created in the last N days. Omit the query to browse the most recent
sessions, with `limit` controlling how many are returned; use `sort_by="recent"`
to order a matching query by recency instead of relevance. All operations are
read-only.

## Copilot editor support

Copilot workspace discovery supports VS Code, VS Code Insiders, Cursor, and
VSCodium on Linux, macOS, and Windows. Use `--config-root` for a non-standard
VS Code installation.

## Development

```bash
uv sync
uv run pytest
```
