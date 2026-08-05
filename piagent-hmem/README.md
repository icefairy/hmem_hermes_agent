# piagent-hmem — HMEM Hybrid Memory Extension for Pi Agent

> **Let your Pi Agent have long-term memory and learn from experience.**

A Pi extension that connects to the HMEM Server to provide:

- 🧠 **Long-term memory** — store and retrieve facts, experiences, and insights
- 🔍 **Hybrid search** — keyword + semantic vector + knowledge graph + time decay
- 🧠 **Reflection engine** — learns mental models from your interactions
- 🕸️ **Knowledge graph** — relationships between memories

## Quick Start

### Prerequisites

1. **HMEM Server** must be running. See [hmem_hermes_agent](https://github.com/icefairy/hmem_hermes_agent) for setup.

### Configuration

Set environment variables (recommended):

```bash
export PIAGENT_HMEM_API_URL="http://localhost:8000"
export PIAGENT_HMEM_API_KEY="your-hmem-api-key"
export PIAGENT_HMEM_NAMESPACE="piagent-default"  # optional, default: "piagent-default"
```

Or create a `.piagent-hmem.json` file in your project root:

```json
{
  "apiUrl": "http://localhost:8000",
  "apiKey": "your-hmem-api-key",
  "namespace": "my-project"
}
```

### Install as Pi Package

```bash
pi install /path/to/piagent-hmem
```

Or copy the extension to your extensions directory:

```bash
cp -r piagent-hmem/extensions ~/.pi/agent/extensions/piagent-hmem
```

### Use via CLI

```bash
pi -e /path/to/piagent-hmem/extensions/index.ts
```

## Available Tools

The extension registers 10 tools for the LLM:

| Tool | Description |
| ------ | ------------- |
| `hmem_write` | Store a memory (content, type, metadata) |
| `hmem_search` | Hybrid semantic search with rerank |
| `hmem_list` | List recent memories, optionally by type |
| `hmem_get` | Get a single memory by ID |
| `hmem_delete` | Delete a memory by ID |
| `hmem_stats` | Memory statistics |
| `hmem_reflect` | Trigger reflection engine |
| `hmem_models` | List mental models and insights |
| `hmem_graph` | Knowledge graph nodes + edges |
| `hmem_namespaces` | List all namespaces |

## Available Commands

| Command | Description |
| --------- | ------------- |
| `/hmem` | Show help |
| `/hmem config` | Show current configuration |
| `/hmem config set <key> <value>` | Set configuration (apiUrl, apiKey, namespace) |
| `/hmem stats` | Memory statistics |
| `/hmem search <query>` | Quick search (limit 5) |
| `/hmem list [type]` | List memories |
| `/hmem models` | List mental models |
| `/hmem reflect` | Trigger reflection |
| `/hmem namespaces` | List namespaces |

## Namespace

Default namespace is `piagent-default`. Each namespace maps to an independent SQLite database in HMEM.

- Same namespace → shared memory across sessions/agents
- Different namespace → complete isolation

## Multi-Agent Support

Multiple Pi agents using the same namespace will share memories and mental models — enabling collaborative learning.

## Architecture

```
Pi Agent ──HTTP──▶ HMEM Server (FastAPI)
                     │
              /api/v1/memories
              /api/v1/search
              /api/v1/stats
              /api/v1/reflect
              /api/v1/mental-models
              /api/v1/graph
              /api/v1/namespaces
                     │
              ┌──────┴──────┐
              │ piagent-    │
              │ default.db  │
              └─────────────┘
```

## Memory Lifecycle

```
Write memories (hmem_write)
        │
        ▼
  Reflection (hmem_reflect)
  ┌─ Accumulate experiences ─┐
  │   LLM clustering         │
  └─→ Insights ─→ Mental Models
        │
        ▼
  Retrieve (hmem_search)
  ←─ Patterns guide behavior
```

## License

MIT
