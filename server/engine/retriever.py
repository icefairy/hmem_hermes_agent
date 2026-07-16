"""Hybrid retriever — combines FTS5 keyword search with vector similarity.

Pipeline:
  1. FTS5 search (jieba-tokenized query)                    → keyword candidates
  2. Vector search (bge-m3 embedding of query)              → semantic candidates
  3. Union + deduplicate candidates
  4. Rerank via rerank_v2_m3 (if available)
  5. Return top-K results

Configurable weights for keyword vs vector contributions.
"""

from __future__ import annotations

import logging
from typing import Any

from .store import HybridMemoryStore
from .embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines FTS5 keyword search with vector similarity search."""

    def __init__(
        self,
        store: HybridMemoryStore,
        embedding_client: EmbeddingClient | None = None,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> None:
        self._store = store
        self._embedding_client = embedding_client
        self._keyword_weight = keyword_weight
        self._vector_weight = vector_weight

    def search(
        self,
        query: str,
        limit: int = 10,
        use_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Hybrid search: FTS5 + vector, optionally reranked.

        Pipeline:
          1. FTS5 keyword search (jieba-tokenized) — get limit*2 candidates
          2. Vector similarity search — get limit*2 candidates
          3. Merge + deduplicate
          4. If rerank enabled and available: rerank with rerank_v2_m3
          5. Return top-K
        """
        if not query or not query.strip():
            return []

        # Stage 1: FTS5 keyword search
        fts_results = self._store.search_fts(
            query, limit=limit * 2
        )

        # Stage 2: Vector search (if embedding client available)
        vec_results: list[dict[str, Any]] = []
        query_embedding: list[float] | None = None
        if self._embedding_client:
            query_embedding = self._embedding_client.embed(query)
            if query_embedding:
                vec_results = self._store.search_vector(
                    query_embedding, limit=limit * 2
                )

        # Stage 3: Merge + deduplicate by memory ID
        merged = self._merge_results(fts_results, vec_results, limit * 3)

        if not merged:
            return []

        # Stage 4: Rerank
        if use_rerank and self._embedding_client and merged:
            documents = [r["content"] for r in merged]
            reranked = self._embedding_client.rerank(query, documents, top_k=limit)
            if reranked and any(r.get("relevance_score", 0) > 0 for r in reranked):
                # Rerank succeeded — reorder by score
                id_map = {r["id"]: r for r in merged}
                final = []
                seen_ids = set()
                for rr in reranked:
                    idx = rr.get("index", -1)
                    if 0 <= idx < len(merged):
                        mem_id = merged[idx]["id"]
                        if mem_id not in seen_ids:
                            entry = id_map[mem_id].copy()
                            entry["score"] = rr.get("relevance_score", 0.0)
                            final.append(entry)
                            seen_ids.add(mem_id)
                merged = final

        # If no rerank scores, compute hybrid scores
        for r in merged:
            if "score" not in r:
                r["score"] = self._compute_score(r, len(merged))

        # Sort by score descending, take top-K
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged[:limit]

    def _merge_results(
        self,
        fts: list[dict[str, Any]],
        vec: list[dict[str, Any]],
        max_candidates: int,
    ) -> list[dict[str, Any]]:
        """Merge FTS and vector results, deduplicating by ID.

        Preserves entries from both sources, combining their metrics.
        """
        seen: set[int] = set()
        merged: list[dict[str, Any]] = []

        # Interleave: highest-ranked from each source
        fts_map = {r["id"]: r for r in fts}
        vec_map = {r["id"]: r for r in vec}

        all_ids: set[int] = set()
        for r in fts:
            all_ids.add(r["id"])
        for r in vec:
            all_ids.add(r["id"])

        for mem_id in all_ids:
            if mem_id in seen:
                continue
            seen.add(mem_id)
            entry: dict[str, Any] = {}
            if mem_id in fts_map:
                entry.update(fts_map[mem_id])
            if mem_id in vec_map:
                # Merge vec fields
                for k, v in vec_map[mem_id].items():
                    if k not in entry:
                        entry[k] = v
            merged.append(entry)

        return merged[:max_candidates]

    def _compute_score(self, entry: dict[str, Any], total: int) -> float:
        """Compute hybrid score from FTS rank and vector similarity."""
        score = 0.0
        total_weight = 0.0

        fts_rank = entry.get("fts_rank")
        if fts_rank is not None:
            # FTS5 rank is negative (lower = better), normalize to [0, 1]
            # rank ≈ -BM25_score, so -rank gives positive BM25-like value
            fts_score = max(0.0, -fts_rank)
            fts_score = 1.0 - 1.0 / (1.0 + fts_score)  # sigmoid-like squash
            score += self._keyword_weight * fts_score
            total_weight += self._keyword_weight

        vec_sim = entry.get("vec_similarity")
        if vec_sim is not None:
            score += self._vector_weight * vec_sim
            total_weight += self._vector_weight

        if total_weight > 0:
            return score / total_weight
        return 0.0