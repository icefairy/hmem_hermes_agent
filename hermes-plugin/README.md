# hermes-plugin · HMEM Memory Provider for Hermes Agent

HMEM hybrid memory — SQLite + sqlite-vec + local HRR holographic memory for
Hermes Agent via the HMEM Server (c/s architecture). This is a thin HTTP client:
no direct DB access, just configure `api_url` + `api_key` and use the memory tools.

> ⚠️ Requirements: a running **HMEM Server**. See [hmem_hermes_agent](https://github.com/icefairy/hmem_hermes_agent) for setup.

## Install

Copy `hermes-plugin/` into your Hermes plugins directory (or point `plugins/experimental`
at it). Register via the Hermes plugin discovery mechanism.

## Configuration (`config.yaml`)

```yaml
plugins:
  hmem:
    api_url: http://hmem-server:8000
    api_key: my-secret-key
    namespace: default
    shared_ns: shared      # optional: tiered shared memory (user prefs / mental models)
```

Environment variables override config.yaml (highest priority):

```bash
export HMEM_API_URL="http://localhost:8000"
export HMEM_API_KEY="my-secret-key"
export HMEM_NAMESPACE="default"
export HMEM_SHARED_NS="shared"      # optional
```

The legacy `plugins.entries.hmem.{...}` structure is also supported.

## Available Tools

| Tool | Description |
| ------ | ------------- |
| `hmem_write` | Store a memory (observation / experience; insight & mental_model are reflect-generated) |
| `hmem_search` | Hybrid search — FTS + semantic vector + local HRR + knowledge-graph expansion; reranked. Params: `query`, `limit`, `min_score` (default 0.1), `namespace`. Merges the `shared_ns` tier at 0.8× |
| `hmem_list` | List recent memories, optionally by type |
| `hmem_get` | Exact read of a single memory by `memory_id` (pure SQL, no vector search) |
| `hmem_delete` | Delete a memory by ID |
| `hmem_stats` | Memory statistics for a namespace |
| `hmem_offload_put` | Offload a large tool result out of context; returns `node_id` + summary |
| `hmem_offload_get` | Restore the full content of an offloaded node by `node_id` |
| `hmem_offload_session` | List offloaded node summaries for a session |
| `hmem_doc_import` | Import a plain-text document into a knowledge base (auto-chunked + vectorized; cascade-overwrite same `doc_id`) |
| `hmem_doc_list` | List documents in a knowledge namespace (with chunk counts) |
| `hmem_doc_get` | Get a document's full content (all chunks) by `doc_id` |
| `hmem_doc_delete` | Cascade-delete a document and all its chunks by `doc_id` |
| `hmem_kb_put` | Add a single knowledge entry (optionally tagged with category/doc_id) |
| `hmem_kb_query` | List knowledge entries, filtered by category/doc_id/tags |
| `hmem_kb_categories` | Knowledge base category summary (entries/documents/tags) |
| `hmem_kb_create` | Create/activate a knowledge base (independent namespace; idempotent) |
| `hmem_kb_list` | List all knowledge bases with entry/doc/category counts |
| `hmem_kb_delete` | Delete a knowledge base (pass `hard=true` to also remove the db file) |

> 🔎 **Knowledge base knowledge** — entries are excluded from time-decay and auto-reflection (fidelity). `hmem_search` hits from KB documents carry a `source` field (`doc_id` / `title` / `chunk_index`) for citation.

## Tiered Shared Memory (multi-role agents)

Role-based isolation by default. Each role keeps its own `namespace` (business
memories + knowledge graph in a closed loop). Set `shared_ns` to merge a shared
tier (user prefs / common mental models / environment layout) into every
`hmem_search` result at 0.8× weight — one place to learn, applied across roles.

```
┌─────────┐ ┌─────────┐ ┌───────┐ ┌────────┐
│ lingjia │ │   dev   │ │  wu   │ │ create │  ← role namespaces (isolated)
└────┬────┘ └────┬────┘ └───┬───┘ └────┬───┘
     └───────────┴─────┬────┴──────────┘
                 ┌─────┴─────┐
                 │  shared   │  ← prefs / mental models (queried at 0.8x)
                 └───────────┘
```

Write shared-tier content explicitly with `namespace: "shared"`.

## Offload pattern (context saving)

```text
1. hmem_offload_put(session_key="task-X", content="<giant-dump>", summary="short")
   → "offloaded to node=abc... refs=/data/.../refs.json"
2. Keep only the summary in context.
3. Later: hmem_offload_get(session_key="task-X", node_id="abc...") → full content
   or hmem_offload_session(session_key="task-X") → list of node summaries.
```

## Version

| Version | Notes |
|---------|-------|
| `1.1.0` | Tiered shared memory (`shared_ns`), `min_score` param, HRR + graph retrieval description |
| `1.0.0` | Initial release: 9 tools (write/search/list/get/delete/stats + offload put/get/session) |
