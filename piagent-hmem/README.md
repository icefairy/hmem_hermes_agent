# pi-hmem — HMEM Hybrid Memory Extension for Pi Agent

> **Let your Pi Agent have long-term memory and learn from experience.**

A Pi extension that connects to the HMEM Server to provide:

- 🧠 **Long-term memory** — store and retrieve facts, experiences, and insights
- 🔍 **Hybrid search** — keyword + semantic vector + knowledge graph + time decay
- 🧠 **Reflection engine** — learns mental models from your interactions
- 🕸️ **Knowledge graph** — relationships between memories

## Install

### From npm (recommended)

```bash
pi install npm:pi-hmem
```

### From source

```bash
pi install /path/to/pi-hmem
```

### Or load directly

```bash
pi -e npm:pi-hmem
pi -e /path/to/pi-hmem/extensions/index.ts
```

## Quick Start

### Prerequisites

1. **HMEM Server** must be running. See [hmem_hermes_agent](https://github.com/icefairy/hmem_hermes_agent) for setup.

### Configuration

Three configuration methods (priority high → low):

#### 1. Environment variables (recommended)

```bash
export PIAGENT_HMEM_API_URL="http://localhost:8000"
export PIAGENT_HMEM_API_KEY="your-hmem-api-key"
export PIAGENT_HMEM_NAMESPACE="piagent"
```

#### 2. Project config file

Create `.pi-hmem.json` in your project root:

```json
{
  "apiUrl": "http://localhost:8000",
  "apiKey": "your-hmem-api-key",
  "namespace": "my-project"
}
```

#### 3. Runtime command

```bash
/hmem config set apiUrl http://localhost:8002
/hmem config set apiKey your-hmem-api-key
/hmem config set namespace piagent
```

## Memory Hierarchy

```
observation ──write──→ experience ──reflect──→ insight ──aggregate──→ mental_model
```

| Type | Description | Write |
| ------ | ------------- | ------- |
| `observation` | Raw facts, user preferences | ✅ `hmem_write` |
| `experience` | Structured experiences (action/context/outcome) | ✅ `hmem_write` |
| `insight` | Patterns discovered by reflection | ❌ Auto-generated only |
| `mental_model` | Abstract behavioral models | ❌ Auto-generated only |

> ⚠️ Only `observation` and `experience` can be written manually. `insight` and `mental_model` are generated exclusively by the `hmem_reflect` reflection engine.

## Available Tools

| Tool | Description |
| ------ | ------------- |
| `hmem_write` | Store a memory (observation or experience) |
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
| `/hmem config set <key> <value>` | Set configuration |
| `/hmem stats` | Memory statistics |
| `/hmem search <query>` | Quick search (limit 5) |
| `/hmem list [type]` | List memories |
| `/hmem models` | List mental models |
| `/hmem reflect` | Trigger reflection |
| `/hmem namespaces` | List namespaces |

## Namespace

Default namespace is `piagent-default`. Each namespace maps to an independent SQLite database.

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
              │ namespace-  │
              │   db.db     │
              └─────────────┘
```

## Memory Lifecycle

```
hmem_write (observation / experience)
        │
        ▼
  hmem_reflect
  ┌─ Accumulate experiences ─┐
  │   LLM clustering         │
  └─→ insights ─→ mental models
        │
        ▼
  hmem_search / auto-prefetch
  ←─ Patterns guide behavior
```

## Memory Prefetch

The extension auto-fetches relevant memories before each turn based on the user's message content, injecting up to 3 relevant memories into the context.

## Version History

| Version | Date | Notes |
|---------|------|-------|
| `0.1.0` | 2026-08-05 | Initial release: 10 tools, 10 commands, reflection integration, auto-prefetch |

## License

MIT
