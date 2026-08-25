"""Hybrid retriever — combines FTS5 keyword search with vector similarity.

Pipeline:
  1. FTS5 search (jieba-tokenized query)                    → keyword candidates
  2. Vector search (bge-m3 embedding of query)              → semantic candidates
  3. Union + deduplicate candidates
  4. Rerank via rerank_v2_m3 (if available)
  5. Filter by min_score, return top-K results

Configurable weights for keyword vs vector contributions.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from engine.embeddings import EmbeddingClient
from engine.store import HybridMemoryStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines FTS5 keyword search with vector similarity search."""

    def __init__(
        self,
        store: HybridMemoryStore,
        embedding_client: EmbeddingClient | None = None,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
        min_score: float | None = None,
        hrr_weight: float = 0.4,
        graph_expand: bool = True,
    ) -> None:
        self._store = store
        self._embedding_client = embedding_client
        self._keyword_weight = keyword_weight
        self._vector_weight = vector_weight
        self._hrr_weight = hrr_weight
        # 最小相关度过滤。None/0 = 不过滤；>0 时纯噪声记忆（如 rerank 分 0.001）会被丢弃
        self._min_score = min_score
        # 图谱扩散：命中记忆沿边补入一跳邻居（联想记忆）
        self._graph_expand = graph_expand

    def search(
        self,
        query: str,
        limit: int = 10,
        use_rerank: bool = True,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search: FTS5 + HRR(本地) + vector, optionally reranked.

        Pipeline:
          1. FTS5 keyword search (jieba-tokenized) — limit*2 candidates
          2. HRR local phase-similarity search (numpy, no API key) — limit*2
          3. Vector similarity search (if embedding client) — limit*2
          4. Merge + deduplicate
          5. If rerank enabled and available: rerank with rerank_v2_m3
          6. Filter by min_score, return top-K

        HRR 是无 API key 场景的语义兑底：任何环境都能跑，写入时自动计算。
        """
        if not query or not query.strip():
            return []

        # Stage 1: FTS5 keyword search
        fts_results = self._store.search_fts(query, limit=limit * 2)

        # Stage 2: HRR local search（本地 numpy，无需 embedding API）
        hrr_results: list[dict[str, Any]] = []
        try:
            if self._hrr_weight > 0:
                from engine.holographic import encode_text, hrr_available

                if hrr_available():
                    dim = self._store.embedding_dim
                    query_vec = encode_text(query, dim)
                    hrr_results = self._store.search_hrr(query_vec, limit=limit * 2)
        except Exception:
            logger.debug("HRR search failed for query: %r", query)

        # Stage 3: Vector search (if embedding client available)
        vec_results: list[dict[str, Any]] = []
        query_embedding: list[float] | None = None
        if self._embedding_client:
            query_embedding = self._embedding_client.embed(query)
            if query_embedding:
                vec_results = self._store.search_vector(
                    query_embedding, limit=limit * 2
                )

        # Stage 4: Merge + deduplicate by memory ID
        merged = self._merge_results(fts_results, hrr_results, vec_results, limit * 3)

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

        # Stage 5: 图谱扩散 — 命中记忆沿边扩展到一跳邻居（联想记忆）
        # 对每个已命中记忆，取其 enriched_to / supporting_evidence 邻居补入上下文。
        # 邻居分压低（0.5×），且标记 graph_expanded，不喧宾夺主。
        if merged and self._graph_expand:
            merged = self._expand_graph(merged, limit)

        # If no rerank scores, compute hybrid scores
        for r in merged:
            if "score" not in r:
                r["score"] = self._compute_score(r)

        # Sort by score descending, take top-K
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # 最小相关度过滤：低于阈值的记忆对当前查询基本无用（如 rerank 的 0.00x），
        # 保留它们只会污染上下文。传 None/0 时不过滤（向后兼容）。
        thr = self._min_score if min_score is None else min_score
        if thr:
            merged = [r for r in merged if r.get("score", 0.0) >= thr]
            if not merged:
                return []

        return merged[:limit]

    def _merge_results(
        self,
        fts: list[dict[str, Any]],
        hrr: list[dict[str, Any]],
        vec: list[dict[str, Any]],
        max_candidates: int,
    ) -> list[dict[str, Any]]:
        """Merge FTS / HRR / vector results, deduplicating by ID.

        Preserves entries from all sources, combining their metrics.
        """
        seen: set[int] = set()
        merged: list[dict[str, Any]] = []

        fts_map = {r["id"]: r for r in fts}
        hrr_map = {r["id"]: r for r in hrr}
        vec_map = {r["id"]: r for r in vec}

        all_ids: set[int] = set()
        for r in fts:
            all_ids.add(r["id"])
        for r in hrr:
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
            if mem_id in hrr_map:
                for k, v in hrr_map[mem_id].items():
                    if k not in entry:
                        entry[k] = v
            if mem_id in vec_map:
                for k, v in vec_map[mem_id].items():
                    if k not in entry:
                        entry[k] = v
            merged.append(entry)

        return merged[:max_candidates]

    def _compute_score(self, entry: dict[str, Any]) -> float:
        """Compute hybrid score from FTS rank, vector similarity, and time decay.

        时间衰减只做“轻量修正”而不是乘法压扁：旧做法把相关度 0.90 的老记忆压到
        0.09，与噪声几乎无法区分，导致 min_score 阈值失效。现在 time_factor
        ∈ [0.775, 1.0]，相关性主导分数，时间仅轻微偏向近期记忆。
        """
        score = 0.0
        total_weight = 0.0

        # Time decay: λ=0.02 (≈35h half-life), clamp to [0, 1]
        # created_at format: "2026-07-16 12:00:00" (CST, UTC+8)
        time_weight = 1.0
        created_at = entry.get("created_at")
        if created_at:
            try:
                from datetime import datetime, timedelta

                ts_clean = created_at.replace("Z", "").replace("+00:00", "")
                t = datetime.fromisoformat(ts_clean)
                # If naive timestamp was UTC, treat as CST for consistent comparison
                if "T" in entry.get("_original_ts", created_at):
                    t = t + timedelta(hours=8)
                hours_ago = (datetime.now() - t).total_seconds() / 3600.0
                time_weight = math.exp(-0.02 * hours_ago)  # λ=0.02
                time_weight = max(
                    0.1, time_weight
                )  # floor at 0.1 (old events still matter)
            except Exception:
                logger.debug("time decay parse failed: %r", created_at)

        # 知识库条目（knowledge 类型或有 doc_id）不随时间衰减——文档知识不看新旧
        if entry.get("memory_type") == "knowledge" or entry.get("doc_id"):
            time_weight = 1.0

        fts_rank = entry.get("fts_rank")
        if fts_rank is not None:
            # FTS5 rank is negative (lower = better), normalize to [0, 1]
            fts_score = max(0.0, -fts_rank)
            fts_score = 1.0 - 1.0 / (1.0 + fts_score)  # sigmoid-like squash
            score += self._keyword_weight * fts_score
            total_weight += self._keyword_weight

        hrr_sim = entry.get("hrr_similarity")
        if hrr_sim is not None:
            # HRR 相位相似度 ∈ [0,1]（无关=0），本地无 API 也可用
            score += self._hrr_weight * hrr_sim
            total_weight += self._hrr_weight

        vec_sim = entry.get("vec_similarity")
        if vec_sim is not None:
            score += self._vector_weight * vec_sim
            total_weight += self._vector_weight

        if total_weight > 0:
            time_factor = 0.75 + 0.25 * time_weight
            return (score / total_weight) * time_factor
        return 0.0

    def _expand_graph(
        self,
        merged: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """图谱扩散：把命中记忆沿边扩展的一跳邻居补入结果。

        目的：让知识图谱在检索中“可用”——命中主记忆 A 时，把 A 的
        enriched_to（上游经验）和 supporting_evidence（下游洞见/模型）邻居
        一并带出，提供联想上下文。

        策略：
          - 只对 top-K 命中记忆查邻居（避免全库扫描）
          - 邻居分 = 0.5 × 主记忆分（不喧宾夺主），并标记 graph_expanded
          - 去重（邻居可能已在本轮结果中）
          - 限制扩散到 limit 条，避免无限增长
        """
        if not merged:
            return merged

        # 只对分数最高的若干命中记忆做扩散（limit=目标数，取前 3 条主记忆）
        hits = sorted(
            merged, key=lambda x: x.get("score", 0.0), reverse=True
        )[: min(3, len(merged))]

        expanded: list[dict[str, Any]] = []
        existing_ids = {r["id"] for r in merged}
        try:
            for hit in hits:
                hit_score = hit.get("score", 0.0)
                neighbors = self._store.get_neighbors(hit["id"], limit=4)
                for nb in neighbors:
                    nid = nb["id"]
                    if nid in existing_ids:
                        continue
                    # 邻居基础分从 hit 分折算，方向/关系影响略降分
                    base = hit_score * 0.5
                    if nb.get("direction") == "in":
                        base *= 0.9
                    entry = {
                        "id": nid,
                        "content": nb["content"],
                        "memory_type": nb.get("memory_type"),
                        "score": base,
                        "graph_expanded": True,
                        "graph_relation": nb.get("relation"),
                        "graph_direction": nb.get("direction"),
                    }
                    expanded.append(entry)
                    existing_ids.add(nid)
                    if len(expanded) >= limit:
                        break
                if len(expanded) >= limit:
                    break
        except Exception:
            logger.debug("graph expansion failed", exc_info=True)

        return merged + expanded
