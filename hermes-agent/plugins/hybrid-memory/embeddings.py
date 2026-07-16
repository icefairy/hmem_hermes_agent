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
                    "input": text,
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
                    "input": texts,
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
                "documents": documents,
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
            logger.debug("Rerank request failed (non-fatal): %s", e)
            # Fallback: return documents with neutral score
            return [
                {"index": i, "relevance_score": 0.0, "content": doc}
                for i, doc in enumerate(documents)
            ]

    def close(self) -> None:
        self._client.close()