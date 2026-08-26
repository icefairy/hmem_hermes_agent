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
export PIAGENT_HMEM_SHARED_NS="shared"   # optional: shared tier memory (user prefs / mental models)
```

#### 2. Project config file

Create `.pi-hmem.json` in your project root:

```json
{
  "apiUrl": "http://localhost:8000",
  "apiKey": "your-hmem-api-key",
  "namespace": "my-project",
  "sharedNs": "shared"
}
```

#### 3. Runtime command

```bash
/hmem config set apiUrl http://localhost:8002
/hmem config set apiKey your-hmem-api-key
/hmem config set namespace piagent
/hmem config set sharedNs shared
```

### Tiered Shared Memory (graded sharing)

Role-based memory isolation by default: each role keeps its own namespace
(business memories + knowledge graph stay self-contained), while a **shared
tier** (`sharedNs`) carries cross-role knowledge — user preferences, common
mental models, environment layout.

- Every `hmem_search` also queries the `sharedNs` namespace (0.8× weight) when set
- Shared hits are tagged `_ns` / `shared: true` in results
- Business memories always go to your own namespace; write shared-tier content
explicitly with `namespace: "shared"`

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ lingjia  │   │   dev    │   │   wu     │   │  create  │  ← role namespaces (isolated, graph closed-loop)
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     └──────────────┴──────┬──────┴──────────────┘
                    ┌──────┴──────┐
                    │   shared    │  ← prefs / mental models (queried at 0.8x)
                    └─────────────┘
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
| `hmem_search` | Hybrid semantic search (FTS + vector + local HRR + graph expand) with rerank; supports `min_score` threshold |
| `hmem_list` | List recent memories, optionally by type |
| `hmem_get` | Get a single memory by ID |
| `hmem_delete` | Delete a memory by ID |
| `hmem_stats` | Memory statistics |
| `hmem_reflect` | Trigger reflection engine |
| `hmem_models` | List mental models and insights |
| `hmem_graph` | Knowledge graph nodes + edges |
| `hmem_namespaces` | List all namespaces |
| `hmem_doc_import` | Import a plain-text document (auto-chunked + vectorized) into a knowledge base |
| `hmem_doc_list` | List documents in a knowledge base |
| `hmem_doc_get` | Get full document content (all chunks) |
| `hmem_doc_delete` | Cascade-delete a document and all its chunks |
| `hmem_kb_put` | Add a single knowledge entry |
| `hmem_kb_query` | List knowledge entries (filter by category/doc_id/tags) |
| `hmem_kb_categories` | Knowledge base category summary |
| `hmem_kb_create` | Create/activate a knowledge base |
| `hmem_kb_list` | List all knowledge bases |
| `hmem_kb_delete` | Delete a knowledge base |

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
| `/hmem kb list` | List knowledge bases |
| `/hmem kb create <ns>` | Create a knowledge base |
| `/hmem kb delete <ns>` | Delete a knowledge base |
| `/hmem doc list` | List documents in default namespace |
| `/hmem doc get <id>` | Get document detail |
| `/hmem doc delete <id>` | Delete document |

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
              /api/v1/documents         ← knowledge base docs
              /api/v1/knowledge          ← knowledge entries
              /api/v1/knowledge-bases    ← library management
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

The extension auto-fetches relevant memories before each turn based on the user's message content, injecting up to 3 relevant memories into the context. When `sharedNs` is configured, shared-tier memories are merged into the results at 0.8× weight.

## Version History

| Version | Date | Notes |
|---------|------|-------|
| `0.3.0` | 2026-08-26 | Knowledge base support: document import/CRUD, knowledge entries, library management (10 new tools) |
| `0.2.1` | 2026-08-25 | README: tiered-shared-memory docs, v0.2.0 notes |
| `0.2.0` | 2026-08-25 | Tiered shared memory (`sharedNs`), `min_score` threshold, HRR local holographic retrieval integration |
| `0.1.0` | 2026-08-05 | Initial release: 10 tools, 10 commands, reflection integration, auto-prefetch |

## License

MIT
