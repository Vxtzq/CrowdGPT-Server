# Using Trail-Code with Self-Hosted Agents (Hermes + llama-server)

Trail-Code is designed primarily for Claude Code, but it works with any agent
because the graft map is just files on disk and the CLI is a standalone tool.
This guide covers wiring it up with a self-hosted Hermes Agent running on
llama-server.

## Three integration levels

### 1. MCP Server (recommended for Hermes)

Trail-Code ships an MCP server that exposes the graph as tools. Hermes has a
built-in MCP client, so this is the cleanest integration.

**Add to `~/.hermes/config.yaml`:**

```yaml
mcp_servers:
  trail:
    command: "trail"
    args: ["mcp", "/data/CrowdGPT-Server"]
    timeout: 120
```

Restart Hermes. You'll get these tools (prefixed with `mcp_trail_`):

| Tool | What it does |
|------|--------------|
| `mcp_trail_graft_find_code` | Find code matching a query, ranked by relevance |
| `mcp_trail_graft_trace_calls` | Trace callers/callees of a symbol |
| `mcp_trail_graft_find_all` | Find all occurrences of a term |
| `mcp_trail_graft_file_api` | Get a file's complete API surface |
| `mcp_trail_graft_repo_map` | Get the repo structure and hotspots |
| `mcp_trail_graft_check_freshness` | Check if the map is up to date |

**Example usage in a conversation:**

```
User: I need to understand how submission verification works before I modify it.
Agent: [calls mcp_trail_graft_find_code "submission verification"]
Agent: Based on the code map, verification.py:45 has the main verify() function.
       It's called by service.py:112 and tested in tests/test_orchestrator_modules.py.
```

**Note:** The MCP server runs in stdio mode. Make sure `trail` is on Hermes's
PATH (it should be if you installed it globally with npm).

### 2. CLI tools via terminal

Hermes can call `trail` commands directly through its terminal tool. This works
without any configuration beyond having `trail` installed.

**Useful commands for agent workflows:**

```bash
# Find relevant code for a task
trail ask "how does the krum aggregation work?"

# Search for a term across the codebase
trail grep "checkpoint"

# See who calls a function (blast radius before editing)
trail callers verify_submission

# Get a file's API surface (fast, ~200 tokens)
trail skeleton orchestrator/submission.py

# Get repo structure and hotspots
trail map

# Check if changes conflict with saved learnings
trail check

# Save a learning after solving something
trail note "The pool persistence layer requires explicit flush after batch inserts"
```

**When to use which:**

| Task | Best command |
|------|--------------|
| "Where is X?" | `trail ask "X"` or `trail grep "X"` |
| "What breaks if I change X?" | `trail callers X` |
| "What's in this file?" | `trail skeleton path/to/file.py` |
| "What changed recently?" | `trail map` |
| "Did I break a convention?" | `trail check` |
| "Remember this for next time" | `trail note "..."` |

### 3. Read the graft map files directly

The graft map is plain markdown files in `.trail/cache/`. Any agent can read
them without Trail-Code installed.

```bash
# Index of all files
cat .trail/cache/INDEX.md

# Specific file's card
cat .trail/cache/orchestrator/submission.md

# Search across all cards
grep -r "checkpoint" .trail/cache/
```

Each card contains a summary of the file, its symbols, and exact file:line
references. This is the fallback when Trail-Code isn't installed but the map
is committed.

## Saving learnings from your agent

The most valuable part of Trail-Code is the learnings. Teach your agent to save
them:

**In your agent's system prompt or project context file:**

```
After solving a non-trivial problem, correcting a mistake, or making a design
decision, save it as a learning:
  trail note "What you learned"
Learnings should be specific, actionable, and useful to someone else working
on this codebase later.
```

**Examples of good learnings:**
- "The pool persistence layer requires explicit flush after batch inserts"
- "Don't use numpy for GPU ops — this repo uses PyTorch exclusively"
- "Checkpoint loading is not thread-safe; serialize with a lock"

**Examples of bad learnings:**
- "Fixed a bug" (too vague)
- "The code is complex" (not actionable)

## Rebuilding the map

The map rebuilds automatically when code changes, but you can force it:

```bash
trail build          # Rebuild the graph
trail build --deep   # Add LLM-generated summaries (needs API key)
trail status         # Check if map is in sync
```

If your agent is working in a branch with significant changes, remind it to
run `trail build` before doing code exploration.

## Project context file

Create a `.hermes.md` or `AGENTS.md` in the repo root to tell any agent about
Trail-Code:

```markdown
# Project Context

This repo uses Trail-Code for code navigation and team learnings.

## Navigation
- Use `trail ask "<task>"` to find relevant code
- Use `trail callers <symbol>` before editing to see blast radius
- Use `trail skeleton <file>` for a quick API overview
- The graft map is in `.trail/cache/` (markdown files)

## Learnings
- Check `.trail/learnings/` for past decisions
- Run `trail check` before committing to catch conflicts
- Save new learnings with `trail note "<insight>"`

## Setup
- Requires Node.js 22+ and `npm install -g @trailhq/trail-code`
- Run `trail build` after cloning to create the local map
```

## Comparison: Claude Code vs self-hosted agent

| Feature | Claude Code (built-in) | Self-hosted (Hermes) |
|---------|----------------------|---------------------|
| Code context injection | Automatic via hooks | Via MCP tools or CLI |
| Blast radius on edit | Automatic | Manual (`trail callers`) |
| Learning capture | Automatic from sessions | Manual (`trail note`) |
| Learning injection | Automatic via hooks | Via MCP tools or CLI |
| Setup | `trail init` | `trail init` + MCP config |

The self-hosted approach requires more explicit calls, but you get full control
over when and how Trail-Code is used. The MCP integration is the closest to the
Claude Code experience.

## Troubleshooting

**MCP tools not appearing in Hermes:**
- Check `~/.hermes/config.yaml` has `mcp_servers` (not `mcp` or `servers`)
- Ensure `trail` is on PATH: `which trail`
- Restart Hermes after config changes
- Check Hermes startup logs for MCP connection messages

**"graft map not found" errors:**
- Run `trail build` in the repo root
- Ensure you're in a git repo with committed code

**Stale map after large changes:**
- Run `trail build` to force rebuild
- Check with `trail status`
