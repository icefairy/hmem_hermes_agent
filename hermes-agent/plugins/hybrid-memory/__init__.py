"""Hybrid memory provider for Hermes Agent.

Combines:
  - SQLite (durable storage)
  - sqlite-vec (vector similarity search)
  - jieba (Chinese tokenization for FTS5)
  - bge-m3 (embedding via provider API)
  - rerank_v2_m3 (reranking via provider API)

Agent spaces:
  Each memory belongs to an ``agent_space`` string. Multiple agents can
  share the same space (``agent_space: "shared"``) or use separate spaces
  (``agent_space: "agent-a"`` vs ``"agent-b"``). The space is configured
  per profile in plugins.hybrid-memory.agent_space.

Config:
  plugins:
    hybrid-memory:
      db_path: "~/.hermes/hybrid_memory.db"     # SQLite database path
      agent_space: "default"                     # shared or per-agent namespace
      embedding_model: "bge-m3"                  # model name for /v1/embeddings
      rerank_model: "rerank_v2_m3"              # model name for /v1/rerank
      embedding_dim: 1024                        # vector dimension
      keyword_weight: 0.4                        # FTS5 weight in hybrid search
      vector_weight: 0.6                        # vector weight in hybrid search
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_PLUGIN_KEY = "hybrid-memory"

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_MEMORY_WRITE_SCHEMA = {
    "name": "hybrid_memory_write",
    "description": (
        "Store a fact in persistent hybrid memory. The fact is indexed with "
        "both FTS5 full-text search (jieba tokenized for Chinese support) and "
        "vector embedding (bge-m3). Use this to remember user preferences, "
        "project decisions, environment details, and any durable information "
        "that should survive across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact content to remember",
            },
            "agent_space": {
                "type": "string",
                "description": "Optional namespace. Defaults to the configured agent_space. "
                               "Use different spaces for multi-agent isolation.",
            },
        },
        "required": ["content"],
    },
}

_MEMORY_READ_SCHEMA = {
    "name": "hybrid_memory_read",
    "description": (
        "Query hybrid memory using combined keyword (FTS5) and semantic (vector) search. "
        "Returns the most relevant memories. Use this to recall what you know "
        "about a person, project, decision, or any topic before answering."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search for",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 10, max: 50)",
            },
            "agent_space": {
                "type": "string",
                "description": "Optional: filter by agent namespace. "
                               "Omit to use configured default.",
            },
        },
        "required": ["query"],
    },
}

_MEMORY_LIST_SCHEMA = {
    "name": "hybrid_memory_list",
    "description": "List recently stored memories. Use for browsing what's in memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max results (default: 20, max: 100)",
            },
            "agent_space": {
                "type": "string",
                "description": "Optional filter by namespace.",
            },
        },
        "required": [],
    },
}

_MEMORY_DELETE_SCHEMA = {
    "name": "hybrid_memory_delete",
    "description": "Delete a memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "description": "ID of the memory to delete",
            },
        },
        "required": ["memory_id"],
    },
}

_MEMORY_STATS_SCHEMA = {
    "name": "hybrid_memory_stats",
    "description": "Get memory statistics (total count, per-space counts).",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    from hermes_cli.config import cfg_get, load_config

    config = load_config()
    return cfg_get(config, "plugins", _PLUGIN_KEY, default={}) or {}


def _read_profile_config(section: str, key: str, default: str = "") -> str:
    """Read a value from config.yaml, trying multiple paths."""
    from hermes_cli.config import cfg_get, load_config
    try:
        config = load_config()
        # Path 1: plugins.hybrid-memory.<key>
        val = cfg_get(config, "plugins", _PLUGIN_KEY, key)
        if val:
            return str(val)
        # Path 2: memory.provider_config.<key>
        val = cfg_get(config, "memory", "provider_config", key)
        if val:
            return str(val)
    except Exception:
        pass
    return default


def _get_base_url() -> str:
    """Get the provider base URL."""
    from hermes_cli.config import cfg_get, load_config
    try:
        config = load_config()
        return str(cfg_get(config, "model", "base_url", default="")) or ""
    except Exception:
        return ""


def _get_api_key() -> str:
    """Get the provider API key from config or env var."""
    from hermes_cli.config import cfg_get, load_config
    import os
    try:
        config = load_config()
        key = str(cfg_get(config, "model", "api_key", default="")) or ""
        # If the key looks truncated (contains ...), check env vars
        if "..." in key:
            for env_var in ["HERMES_EMBEDDING_API_KEY", "OPENAI_API_KEY", "VLLM_API_KEY"]:
                ek = os.environ.get(env_var, "")
                if ek:
                    return ek
        return key
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------


class HybridMemoryProvider(MemoryProvider):
    """Hybrid memory provider with SQLite + sqlite-vec + jieba + bge-m3."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._embedding_client = None
        self._agent_space = self._config.get("agent_space", "default") or "default"
        self._initialized = False

    @property
    def name(self) -> str:
        return "hybrid-memory"

    def is_available(self) -> bool:
        """Check dependencies are installed."""
        try:
            import jieba  # noqa: F401
            import sqlite_vec  # noqa: F401
            return True
        except ImportError:
            return False

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/hybrid_memory.db"
        return [
            {
                "key": "db_path",
                "description": "SQLite database path",
                "default": _default_db,
            },
            {
                "key": "agent_space",
                "description": "Namespace for memory isolation (shared or per-agent)",
                "default": "default",
            },
            {
                "key": "embedding_model",
                "description": "Model name for /v1/embeddings (e.g. bge-m3)",
                "default": "bge-m3",
            },
            {
                "key": "rerank_model",
                "description": "Model name for /v1/rerank (e.g. rerank_v2_m3)",
                "default": "rerank_v2_m3",
            },
            {
                "key": "embedding_dim",
                "description": "Vector dimension (default: 1024)",
                "default": "1024",
            },
            {
                "key": "keyword_weight",
                "description": "FTS5 keyword search weight in hybrid scoring (0-1)",
                "default": "0.4",
            },
            {
                "key": "vector_weight",
                "description": "Vector search weight in hybrid scoring (0-1)",
                "default": "0.6",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"][_PLUGIN_KEY] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.warning("Failed to save hybrid-memory config: %s", e)

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = f"{_hermes_home}/hybrid_memory.db"

        # Resolve paths
        db_path = self._config.get("db_path", _default_db)
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)

        embedding_dim = int(self._config.get("embedding_dim", 1024))
        embedding_model = self._config.get("embedding_model", "bge-m3")
        rerank_model = self._config.get("rerank_model", "rerank_v2_m3")
        keyword_weight = float(self._config.get("keyword_weight", 0.4))
        vector_weight = float(self._config.get("vector_weight", 0.6))

        # Override from config.yaml top-level
        cfg_embed_model = _read_profile_config("embedding_model", "")
        cfg_rerank_model = _read_profile_config("rerank_model", "")
        if cfg_embed_model:
            embedding_model = cfg_embed_model
        if cfg_rerank_model:
            rerank_model = cfg_rerank_model

        # Read base_url / api_key from profile config (with env fallback)
        base_url = _get_base_url()
        api_key = _get_api_key()

        # Plugin-specific overrides (highest priority)
        plugin_embed_url = self._config.get("embedding_base_url", "")
        plugin_embed_key = self._config.get("embedding_api_key", "")
        if plugin_embed_url:
            base_url = plugin_embed_url
        if plugin_embed_key:
            api_key = plugin_embed_key

        # Initialize store
        from .store import HybridMemoryStore
        self._store = HybridMemoryStore(
            db_path=db_path,
            embedding_dim=embedding_dim,
        )
        self._store.initialize()

        # Initialize embedding client
        if base_url and api_key:
            from .embeddings import EmbeddingClient
            self._embedding_client = EmbeddingClient(
                base_url=base_url,
                api_key=api_key,
                embedding_model=embedding_model,
                rerank_model=rerank_model,
                embedding_dim=embedding_dim,
            )
        else:
            logger.warning(
                "No base_url/api_key configured — embeddings and rerank disabled."
            )

        # Initialize retriever
        from .retriever import HybridRetriever
        self._retriever = HybridRetriever(
            store=self._store,
            embedding_client=self._embedding_client,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
        )

        self._agent_space = self._config.get("agent_space", "default") or "default"
        self._session_id = session_id
        self._initialized = True

        logger.info(
            "HybridMemory initialized: db=%s space=%s embed=%s rerank=%s dim=%d",
            db_path, self._agent_space, embedding_model, rerank_model, embedding_dim,
        )

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        total = self._store.count_memories()
        space_total = self._store.count_memories(agent_space=self._agent_space)
        if total == 0:
            return (
                "# Hybrid Memory\n"
                "Active (SQLite + sqlite-vec + jieba + bge-m3). "
                "Empty store — proactively add facts the user expects you to remember.\n"
                "Commands: hybrid_memory_write (store), hybrid_memory_read (search), "
                "hybrid_memory_list (browse), hybrid_memory_delete (remove)."
            )
        return (
            f"# Hybrid Memory\n"
            f"Active. {total} total memories ({space_total} in current space).\n"
            "Hybrid search: FTS5 keyword (jieba Chinese tokenization) + bge-m3 vector "
            "similarity, reranked via rerank_v2_m3.\n"
            "Commands: hybrid_memory_write, hybrid_memory_read, hybrid_memory_list, "
            "hybrid_memory_delete, hybrid_memory_stats."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Prefetch relevant memories for system prompt context."""
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(
                query,
                agent_space=self._agent_space,
                limit=5,
                use_rerank=True,
            )
            if not results:
                return ""
            lines = []
            for r in results:
                score = r.get("score", 0.0)
                lines.append(f"- [{score:.2f}] {r['content']}")
            return "## Hybrid Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Hybrid prefetch failed: %s", e)
            return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Auto-extract key facts from turns (if configured)."""
        # Auto-extraction is handled by on_session_end if enabled
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            _MEMORY_WRITE_SCHEMA,
            _MEMORY_READ_SCHEMA,
            _MEMORY_LIST_SCHEMA,
            _MEMORY_DELETE_SCHEMA,
            _MEMORY_STATS_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        handlers = {
            "hybrid_memory_write": self._handle_write,
            "hybrid_memory_read": self._handle_read,
            "hybrid_memory_list": self._handle_list,
            "hybrid_memory_delete": self._handle_delete,
            "hybrid_memory_stats": self._handle_stats,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return tool_error(f"Unknown tool: {tool_name}")
        return handler(args)

    def shutdown(self) -> None:
        if self._embedding_client:
            self._embedding_client.close()
        if self._store:
            self._store.close()
        self._store = None
        self._retriever = None
        self._embedding_client = None
        self._initialized = False

    # -- Tool handlers -------------------------------------------------------

    def _handle_write(self, args: dict) -> str:
        """Store a fact with jieba + FTS5 indexing and bge-m3 embedding."""
        if not self._store:
            return tool_error("Memory store not initialized")
        try:
            content = args["content"]
            agent_space = args.get("agent_space", self._agent_space)

            # Get embedding if client available
            embedding = None
            if self._embedding_client:
                embedding = self._embedding_client.embed(content)

            memory_id = self._store.add_memory(content, agent_space, embedding)
            if memory_id is None:
                return tool_error("Failed to store memory")
            return json.dumps({
                "memory_id": memory_id,
                "content": content[:100] + ("..." if len(content) > 100 else ""),
                "agent_space": agent_space,
                "embedded": embedding is not None,
            })
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_read(self, args: dict) -> str:
        """Hybrid search: FTS5 + vector + rerank."""
        if not self._retriever:
            return tool_error("Memory retriever not initialized")
        try:
            query = args["query"]
            limit = min(int(args.get("limit", 10)), 50)
            agent_space = args.get("agent_space", self._agent_space)

            results = self._retriever.search(
                query, agent_space=agent_space, limit=limit, use_rerank=True
            )
            return json.dumps({"results": results, "count": len(results)})
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_list(self, args: dict) -> str:
        """List recent memories."""
        if not self._store:
            return tool_error("Memory store not initialized")
        try:
            limit = min(int(args.get("limit", 20)), 100)
            agent_space = args.get("agent_space", self._agent_space)
            results = self._store.list_memories(
                agent_space=agent_space, limit=limit
            )
            return json.dumps({"results": results, "count": len(results)})
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_delete(self, args: dict) -> str:
        """Delete a memory by ID."""
        if not self._store:
            return tool_error("Memory store not initialized")
        try:
            memory_id = int(args["memory_id"])
            success = self._store.delete_memory(memory_id)
            return json.dumps({"deleted": success, "memory_id": memory_id})
        except (KeyError, ValueError) as exc:
            return tool_error(f"Invalid argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_stats(self, args: dict) -> str:
        """Return memory statistics."""
        if not self._store:
            return tool_error("Memory store not initialized")
        try:
            total = self._store.count_memories()
            space_total = self._store.count_memories(
                agent_space=self._agent_space
            )
            return json.dumps({
                "total_memories": total,
                "current_space": self._agent_space,
                "current_space_count": space_total,
                "embedding_enabled": self._embedding_client is not None,
            })
        except Exception as exc:
            return tool_error(str(exc))


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the hybrid memory provider."""
    config = _load_plugin_config()
    provider = HybridMemoryProvider(config=config)
    ctx.register_memory_provider(provider)