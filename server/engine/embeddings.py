"""Embedding and reranking API client for hybrid memory.

Calls bge-m3 (embedding) and rerank_v2_m3 (reranking) through
the provider's OpenAI-compatible API endpoint.

Config resolution order:
1. plugin config ``plugins.hybrid-memory.{embedding_model, rerank_model}``
2. ``memory.provider_config.{embedding_model, rerank_model}``
3. Fallback to ``bge-m3`` / ``rerank_v2_m3``

The API base URL and auth key are inherited from the profile's
``model.base_url`` / ``model.api_key`` in config.yaml.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default timeouts (seconds)
_EMBED_TIMEOUT = 30.0
_RERANK_TIMEOUT = 30.0

# 上游上下文硬限保护（2026-09-16 实测硅基流动）：
#   rerank: 单篇 document 超 ~17000 字符 → 整个请求 400
#           "This model's maximum context length is 8192 tokens"
#   embed:  单条 input 超 ~8192 token → 400 "The parameter is invalid"
# 写入侧已对超长内容分片（store._SHARD_CHARS=3000），这里是存量数据与
# 未走分片路径调用（scripts/dedup 等）的兼容兵：超长则截断，
# 宁可语义略损，也不要整个请求失败（rerank 失败会静默退化为 0 分排序）。
_MAX_DOC_CHARS = 3000
_MAX_EMBED_INPUT_CHARS = 6000


def _clip(text: str, max_chars: int, label: str) -> str:
    """截断超长文本到上游可接受范围（截断时记 warning，不再静默）。"""
    if not isinstance(text, str) or len(text) <= max_chars:
        return text
    logger.warning(
        "%s too long, truncating %d -> %d chars (upstream context limit)",
        label,
        len(text),
        max_chars,
    )
    return text[:max_chars]


class EmbeddingClient:
    """OpenAI-compatible embedding and reranking client.

    Uses the same base_url and api_key as the profile's model config,
    so no separate credential setup is needed.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        embedding_model: str = "bge-m3",
        rerank_model: str = "rerank_v2_m3",
        embedding_dim: int = 1024,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model
        self._embedding_dim = embedding_dim
        self._client = httpx.Client(timeout=_EMBED_TIMEOUT)

    def embed(self, text: str) -> list[float] | None:
        """Get embedding vector for a single text string.

        Returns a list of floats (dimension = embedding_dim), or None on failure.
        """
        if not text or not text.strip():
            return None
        try:
            resp = self._client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._embedding_model,
                    "input": _clip(text, _MAX_EMBED_INPUT_CHARS, "embedding input"),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            # Truncate or pad to expected dimension
            if len(embedding) > self._embedding_dim:
                embedding = embedding[: self._embedding_dim]
            elif len(embedding) < self._embedding_dim:
                embedding = embedding + [0.0] * (self._embedding_dim - len(embedding))
            return embedding
        except Exception as e:
            logger.warning("Embedding request failed: %s", e)
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Get embedding vectors for a batch of texts.

        Returns list of embedding vectors (or None per item on failure).
        Batch size is unbounded — provider may truncate; caller should
        chunk to reasonable sizes (e.g. 32) for production use.
        """
        if not texts:
            return []
        try:
            resp = self._client.post(
                f"{self._base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._embedding_model,
                    "input": [
                        _clip(t, _MAX_EMBED_INPUT_CHARS, "embedding input") for t in texts
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            # Map back by index — some providers reorder; most return in order
            by_idx: dict[int, list[float]] = {}
            for item in data["data"]:
                idx = item.get("index", len(by_idx))
                embedding = item["embedding"]
                if len(embedding) > self._embedding_dim:
                    embedding = embedding[: self._embedding_dim]
                elif len(embedding) < self._embedding_dim:
                    embedding = embedding + [0.0] * (self._embedding_dim - len(embedding))
                by_idx[idx] = embedding
            return [by_idx.get(i) for i in range(len(texts))]
        except Exception as e:
            logger.warning("Batch embedding request failed: %s", e)
            return [None] * len(texts)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents by relevance to query.

        Returns list of dicts:
          {"index": int, "relevance_score": float, "content": str}

        If rerank endpoint is unavailable, returns documents with
        default score of 0.0 (fallback — caller can use pre-rerank order).
        """
        if not documents:
            return []
        try:
            body: dict[str, Any] = {
                "model": self._rerank_model,
                "query": query,
                # 单篇超长会让整个 rerank 请求 400（上游按 query+单篇合计限 8192 token），
                # 逐篇截断保护存量未分片数据。
                "documents": [
                    _clip(d, _MAX_DOC_CHARS, "rerank document") for d in documents
                ],
            }
            if top_k is not None:
                body["top_k"] = top_k

            resp = self._client.post(
                f"{self._base_url}/rerank",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=_RERANK_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            # Attach content for convenience
            for r in results:
                idx = r.get("index", -1)
                if 0 <= idx < len(documents):
                    r["content"] = documents[idx]
            return results
        except Exception as e:
            logger.warning("Rerank request failed (non-fatal): %s", e)
            # Fallback: return documents with neutral score
            return [
                {"index": i, "relevance_score": 0.0, "content": doc}
                for i, doc in enumerate(documents)
            ]

    def close(self) -> None:
        self._client.close()