"""
HMEM Hermes Plugin — 轻量 API 客户端。

通过 HTTP 调用 HMEM Server，实现记忆存储/检索/管理。
无需直接访问数据库，只需配置 api_url + api_key 即可使用。

配置示例 (config.yaml):
  plugins:
    hmem:
      api_url: http://hmem-server:8000
      api_key: my-secret-key
      namespace: default
      shared_ns: shared    # 可选：分级共享记忆库（检索时一并查询，权重 x0.8）
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# 尝试导入 Hermes 接口（插件模式下可选）
try:
    from agent.memory_provider import MemoryProvider
    from tools.registry import tool_error
    _HAS_HERMES = True
except ImportError:
    _HAS_HERMES = False
    # 独立使用时的兜底
    MemoryProvider = object  # type: ignore


_PLUGIN_KEY = "hmem"

_MEMORY_WRITE_SCHEMA = {
    "name": "hmem_write",
    "description": "Store a fact in HMEM memory server. Supports structured fields: action, context, outcome.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact content to remember"},
            "namespace": {"type": "string", "description": "Optional namespace for multi-agent isolation"},
            "memory_type": {"type": "string", "description": "Type: observation (raw), experience (structured with action/outcome), insight, mental_model"},
            "mem_action": {"type": "string", "description": "Action type (code_generation, qa, debug, ...)"},
        },
        "required": ["content"],
    },
}

_MEMORY_READ_SCHEMA = {
    "name": "hmem_search",
    "description": (
        "Hybrid search across memories using keywords + semantic similarity + "
        "local HRR holographic + knowledge graph expansion (via HMEM server). "
        "Optionally also queries the shared namespace (shared_ns) at 0.8x weight."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language query"},
            "limit": {"type": "integer", "description": "Max results (default: 10, max: 50)"},
            "min_score": {"type": "number", "description": "Min relevance threshold (default: server 0.1; 0 = keep all noise)"},
            "namespace": {"type": "string", "description": "Optional filter by namespace"},
        },
        "required": ["query"],
    },
}

_MEMORY_LIST_SCHEMA = {
    "name": "hmem_list",
    "description": "List recently stored memories.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max results (default: 20, max: 100)"},
            "memory_type": {"type": "string", "description": "Filter: observation, experience, insight, mental_model"},
            "namespace": {"type": "string", "description": "Optional namespace filter"},
        },
        "required": [],
    },
}

_MEMORY_DELETE_SCHEMA = {
    "name": "hmem_delete",
    "description": "Delete a memory by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer", "description": "ID of the memory to delete"},
        },
        "required": ["memory_id"],
    },
}

_MEMORY_GET_SCHEMA = {
    "name": "hmem_get",
    "description": (
        "Exact read of a single memory by memory_id (pure SQL, no vector search). "
        "Use this when you already know the memory_id — faster and more accurate "
        "than hmem_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer", "description": "ID of the memory to read"},
            "namespace": {"type": "string", "description": "Optional namespace override"},
        },
        "required": ["memory_id"],
    },
}

_MEMORY_STATS_SCHEMA = {
    "name": "hmem_stats",
    "description": "Get memory statistics from HMEM server.",
    "parameters": {"type": "object", "properties": {
        "namespace": {"type": "string", "description": "Optional namespace filter"},
    }, "required": []},
}

# ── Context Offloading tools ──────────────────────────────────────────

_OFFLOAD_PUT_SCHEMA = {
    "name": "hmem_offload_put",
    "description": (
        "Offload a large tool result / long text out of the LLM context. "
        "Stores the full original content in HMEM (100% retrievable) and keeps "
        "only a short summary in context. Use when a tool result is too long to "
        "keep inline (e.g. huge file dumps, long command output). Returns "
        "node_id + summary to reference the content later."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_key": {"type": "string", "description": "Session identifier (e.g. current task id)"},
            "content": {"type": "string", "description": "The full original content to offload"},
            "node_id": {"type": "string", "description": "Optional stable node id; auto-generated if omitted"},
            "summary": {"type": "string", "description": "Optional short summary (<=200 chars). Auto-generated one-line if omitted"},
            "content_type": {"type": "string", "description": "Content type hint, default 'text'"},
            "namespace": {"type": "string", "description": "Optional namespace override"},
        },
        "required": ["session_key", "content"],
    },
}

_OFFLOAD_GET_SCHEMA = {
    "name": "hmem_offload_get",
    "description": "Retrieve the full original content of an offloaded node by node_id (restore what was offloaded earlier).",
    "parameters": {
        "type": "object",
        "properties": {
            "session_key": {"type": "string", "description": "Session identifier"},
            "node_id": {"type": "string", "description": "node_id returned by hmem_offload_put"},
            "namespace": {"type": "string", "description": "Optional namespace override"},
        },
        "required": ["session_key", "node_id"],
    },
}

_OFFLOAD_SESSION_SCHEMA = {
    "name": "hmem_offload_session",
    "description": (
        "List all offloaded summaries (node_id + summary, no full content) for a "
        "session. Use to re-inject compact context about a long task, or to find "
        "which node to restore via hmem_offload_get."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_key": {"type": "string", "description": "Session identifier"},
            "namespace": {"type": "string", "description": "Optional namespace override"},
        },
        "required": ["session_key"],
    },
}


class HmemMemoryProvider(MemoryProvider):
    """HMEM API 客户端 — 通过 HTTP 调用远程记忆服务。"""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._api_url = (self._config.get("api_url") or "http://localhost:8000").rstrip("/")
        self._api_key = self._config.get("api_key", "")
        self._namespace = self._config.get("namespace", "default") or "default"
        # 分级共享：额外检索的共享库名（如 "shared"），存放用户偏好/通用心智模型
        self._shared_ns = (self._config.get("shared_ns") or "").strip()
        self._http: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "hmem"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._http = httpx.Client(base_url=self._api_url, timeout=60.0)
        # 探活
        try:
            r = self._http.get("/health")
            r.raise_for_status()
            logger.info("HMEM server connected: %s", self._api_url)
        except Exception as e:
            logger.warning("HMEM server not reachable at %s: %s", self._api_url, e)
        self._session_id = session_id

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        """调用 HMEM API。"""
        if not self._http:
            return {"error": "not_initialized"}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            if method == "GET":
                resp = self._http.get(path, params=body, headers=headers)
            else:
                resp = self._http.request(method, path, json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HMEM API {e.response.status_code}: {e.response.text[:200]}"}
        except Exception as e:
            return {"error": f"HMEM request failed: {e}"}

    def system_prompt_block(self) -> str:
        stats = self._call("GET", f"/api/v1/stats?namespace={self._namespace}")
        total = stats.get("total_memories", "?")
        embed = stats.get("embedding_enabled", False)
        return (
            f"# HMEM Memory\n"
            f"Connected to HMEM server ({self._api_url}). "
            f"{total} memories, embeddings {'ON' if embed else 'OFF'}. "
            f"Tools: hmem_write, hmem_search, hmem_get, hmem_list, hmem_delete, hmem_stats, "
            f"hmem_offload_put, hmem_offload_get, hmem_offload_session."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query:
            return ""
        body = {
            "query": query, "namespace": self._namespace, "limit": 5, "use_rerank": True,
        }
        if self._shared_ns and self._shared_ns != self._namespace:
            body["extra_namespaces"] = [self._shared_ns]
        result = self._call("POST", "/api/v1/search", body)
        items = result.get("results", [])
        if not items:
            return ""
        lines = [f"- [{r.get('score', 0):.2f}] {r['content']}" for r in items]
        return "## HMEM Memory\n" + "\n".join(lines)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            _MEMORY_WRITE_SCHEMA, _MEMORY_READ_SCHEMA,
            _MEMORY_LIST_SCHEMA, _MEMORY_DELETE_SCHEMA, _MEMORY_GET_SCHEMA,
            _MEMORY_STATS_SCHEMA,
            _OFFLOAD_PUT_SCHEMA, _OFFLOAD_GET_SCHEMA, _OFFLOAD_SESSION_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        handlers = {
            "hmem_write": self._handle_write,
            "hmem_search": self._handle_search,
            "hmem_list": self._handle_list,
            "hmem_delete": self._handle_delete,
            "hmem_get": self._handle_get,
            "hmem_stats": self._handle_stats,
            "hmem_offload_put": self._handle_offload_put,
            "hmem_offload_get": self._handle_offload_get,
            "hmem_offload_session": self._handle_offload_session,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return tool_error(f"Unknown tool: {tool_name}") if _HAS_HERMES else f"error: unknown tool {tool_name}"
        return handler(args)

    # ── Context Offloading handlers ────────────────────────────────

    def _handle_offload_put(self, args: Dict[str, Any]) -> str:
        ns = args.get("namespace") or self._namespace
        body = {
            "session_key": args.get("session_key", ""),
            "content": args.get("content", ""),
            "content_type": args.get("content_type", "text"),
            "namespace": ns,
        }
        if args.get("node_id"):
            body["node_id"] = args["node_id"]
        if args.get("summary"):
            body["summary"] = args["summary"]
        result = self._call("POST", "/api/v1/offload/put", body)
        if "error" in result:
            return result["error"]
        node = result.get("node_id", "?")
        summary = result.get("summary", "")
        refs = result.get("refs_path", "")
        return f"offloaded to node={node} refs={refs}\nsummary: {summary}\n(use hmem_offload_get session_key={args.get('session_key')} node_id={node} to restore the full content)"

    def _handle_offload_get(self, args: Dict[str, Any]) -> str:
        ns = args.get("namespace") or self._namespace
        result = self._call("GET", "/api/v1/offload/get", {
            "session_key": args.get("session_key", ""),
            "node_id": args.get("node_id", ""),
            "namespace": ns,
        })
        if "error" in result:
            return result["error"]
        return result.get("content", "")

    def _handle_offload_session(self, args: Dict[str, Any]) -> str:
        ns = args.get("namespace") or self._namespace
        session_key = args.get("session_key", "")
        result = self._call("GET", f"/api/v1/offload/session/{session_key}", {
            "namespace": ns,
        })
        if "error" in result:
            return result["error"]
        records = result.get("records", [])
        if not records:
            return "no offloaded nodes for this session"
        lines = [f"- {r['node_id']} ({r['content_type']}): {r['summary']}" for r in records]
        return f"offload session {session_key} ({result.get('count', len(records))} nodes):\n" + "\n".join(lines)

    def _handle_write(self, args: dict) -> str:
        body = {
            "content": args["content"],
            "namespace": args.get("namespace", self._namespace),
            "memory_type": args.get("memory_type", "observation"),
            "mem_action": args.get("mem_action", ""),
        }
        result = self._call("POST", "/api/v1/memories", body)
        if "error" in result:
            return tool_error(result["error"]) if _HAS_HERMES else json.dumps(result)
        return json.dumps({
            "memory_id": result.get("memory_id"),
            "content": result.get("content", ""),
            "namespace": result.get("namespace", ""),
            "embedded": result.get("embedded", False),
        })

    def _handle_search(self, args: dict) -> str:
        body = {
            "query": args["query"],
            "limit": min(int(args.get("limit", 10)), 50),
            "namespace": args.get("namespace", self._namespace),
            "use_rerank": True,
        }
        if "min_score" in args and args["min_score"] is not None:
            body["min_score"] = float(args["min_score"])
        if self._shared_ns and self._shared_ns != body["namespace"]:
            body["extra_namespaces"] = [self._shared_ns]
        result = self._call("POST", "/api/v1/search", body)
        items = result.get("results", [])
        return json.dumps({"results": items, "count": len(items)})

    def _handle_list(self, args: dict) -> str:
        params = {
            "namespace": args.get("namespace", self._namespace),
            "limit": min(int(args.get("limit", 20)), 100),
            "memory_type": args.get("memory_type"),
        }
        result = self._call("GET", "/api/v1/memories", params)
        items = result.get("results", [])
        return json.dumps({"results": items, "count": len(items)})

    def _handle_delete(self, args: dict) -> str:
        mid = int(args["memory_id"])
        ns = args.get("namespace", self._namespace)
        result = self._call("DELETE", f"/api/v1/memories/{mid}?namespace={ns}")
        return json.dumps(result)

    def _handle_get(self, args: dict) -> str:
        mid = int(args["memory_id"])
        ns = args.get("namespace", self._namespace)
        result = self._call("GET", f"/api/v1/memories/{mid}?namespace={ns}")
        if "error" in result:
            return result["error"]
        return json.dumps(result)

    def _handle_stats(self, args: dict) -> str:
        ns = args.get("namespace", self._namespace)
        result = self._call("GET", f"/api/v1/stats?namespace={ns}")
        if "error" in result:
            return tool_error(result["error"]) if _HAS_HERMES else json.dumps(result)
        return json.dumps(result)

    def shutdown(self) -> None:
        if self._http:
            self._http.close()
            self._http = None


# 注册入口（Hermes 插件发现机制）
def register(ctx) -> None:
    """Register the HMEM memory provider."""
    # 从环境变量读取（优先级最高），回退到 config.yaml
    #   HMEM_API_URL - HMEM 服务器地址（默认 http://localhost:8000）
    #   HMEM_API_KEY - HMEM API Key
    #   HMEM_NAMESPACE - 默认 namespace
    #   HMEM_SHARED_NS - 分级共享记忆库（可选，检索时一并查询）
    import os
    env_api_url = os.environ.get("HMEM_API_URL", "").strip()
    env_api_key = os.environ.get("HMEM_API_KEY", "").strip()
    env_namespace = os.environ.get("HMEM_NAMESPACE", "").strip()
    env_shared_ns = os.environ.get("HMEM_SHARED_NS", "").strip()

    # 从 Hermes config 读 hmem 插件配置
    # 兼容两种结构：
    #   原: plugins.hmem.{api_url, api_key, namespace}
    #   新: plugins.entries.hmem.{...}
    try:
        from hermes_cli.config import cfg_get, load_config
        config = load_config()
        plugin_cfg = cfg_get(config, "plugins", _PLUGIN_KEY, default={}) or {}
        if not plugin_cfg:
            entries = cfg_get(config, "plugins", "entries", default={}) or {}
            plugin_cfg = entries.get(_PLUGIN_KEY, {}) or {}
    except Exception:
        plugin_cfg = {}

    # 环境变量覆盖 config 中的值
    if env_api_url:
        plugin_cfg["api_url"] = env_api_url
    if env_api_key:
        plugin_cfg["api_key"] = env_api_key
    if env_namespace:
        plugin_cfg["namespace"] = env_namespace
    if env_shared_ns:
        plugin_cfg["shared_ns"] = env_shared_ns

    provider = HmemMemoryProvider(config=plugin_cfg)
    ctx.register_memory_provider(provider)
