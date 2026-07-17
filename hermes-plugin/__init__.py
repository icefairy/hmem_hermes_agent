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
    "description": "Hybrid search across memories using keywords + semantic similarity (via HMEM server).",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language query"},
            "limit": {"type": "integer", "description": "Max results (default: 10, max: 50)"},
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
            "memory_type": {"type": "string", "description": "Filter by type: experience, fact, mental_model"},
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

_MEMORY_STATS_SCHEMA = {
    "name": "hmem_stats",
    "description": "Get memory statistics from HMEM server.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class HmemMemoryProvider(MemoryProvider):
    """HMEM API 客户端 — 通过 HTTP 调用远程记忆服务。"""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._api_url = (self._config.get("api_url") or "http://localhost:8000").rstrip("/")
        self._api_key = self._config.get("api_key", "")
        self._namespace = self._config.get("namespace", "default") or "default"
        self._http: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "hmem"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._http = httpx.Client(base_url=self._api_url, timeout=15.0)
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
            f"Tools: hmem_write, hmem_search, hmem_list, hmem_delete, hmem_stats."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query:
            return ""
        result = self._call("POST", "/api/v1/search", {
            "query": query, "namespace": self._namespace, "limit": 5, "use_rerank": True,
        })
        items = result.get("results", [])
        if not items:
            return ""
        lines = [f"- [{r.get('score', 0):.2f}] {r['content']}" for r in items]
        return "## HMEM Memory\n" + "\n".join(lines)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            _MEMORY_WRITE_SCHEMA, _MEMORY_READ_SCHEMA,
            _MEMORY_LIST_SCHEMA, _MEMORY_DELETE_SCHEMA, _MEMORY_STATS_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        handlers = {
            "hmem_write": self._handle_write,
            "hmem_search": self._handle_search,
            "hmem_list": self._handle_list,
            "hmem_delete": self._handle_delete,
            "hmem_stats": self._handle_stats,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return tool_error(f"Unknown tool: {tool_name}") if _HAS_HERMES else f"error: unknown tool {tool_name}"
        return handler(args)

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
        result = self._call("POST", "/api/v1/search", body)
        items = result.get("results", [])
        return json.dumps({"results": items, "count": len(items)})

    def _handle_list(self, args: dict) -> str:
        params = {
            "limit": min(int(args.get("limit", 20)), 100),
            "memory_type": args.get("memory_type"),
        }
        result = self._call("GET", "/api/v1/memories", params)
        items = result.get("results", [])
        return json.dumps({"results": items, "count": len(items)})

    def _handle_delete(self, args: dict) -> str:
        mid = int(args["memory_id"])
        result = self._call("DELETE", f"/api/v1/memories/{mid}")
        return json.dumps(result)

    def _handle_stats(self, args: dict) -> str:
        result = self._call("GET", f"/api/v1/stats?namespace={self._namespace}")
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
    # 从 Hermes config 读 hmem 插件配置
    try:
        from hermes_cli.config import cfg_get, load_config
        config = load_config()
        plugin_cfg = cfg_get(config, "plugins", _PLUGIN_KEY, default={}) or {}
    except Exception:
        plugin_cfg = {}
    provider = HmemMemoryProvider(config=plugin_cfg)
    ctx.register_memory_provider(provider)