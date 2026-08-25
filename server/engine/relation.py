"""组合推理 — reason / contrad 运算。

基于 HRR 相位向量的组合检索：
  reason(entities)  — 向量空间 JOIN：找同时与多个实体相关的记忆
  contradict        — 记忆卫生：找共享实体但内容低相似（可能互相矛盾）的记忆

这些能力是纯本地 numpy 实现的，无模型 API 依赖，与第③层检索链同时可用。
"""

from __future__ import annotations

import logging
from typing import Any

from engine.holographic import (
    bytes_to_phases,
    encode_text,
    similarity,
)
from engine.store import HybridMemoryStore

logger = logging.getLogger(__name__)

# 与 store.py 保持一致的 Sqlite 表名
_MAIN_TABLE = "memories"
_HRR_TABLE = "hrr_memories"


def _all_hrr_memories(store: HybridMemoryStore) -> list[dict[str, Any]]:
    """读取全部带 HRR 向量的记忆（id, content, memory_type, created_at, 向量）。"""
    rows = store._conn.execute(
        f"SELECT m.id, m.content, m.memory_type, m.created_at, "
        f"  m.updated_at, h.hrr_vector, h.dim "
        f"FROM {_MAIN_TABLE} m "
        f"JOIN {_HRR_TABLE} h ON h.memory_id = m.id"
    ).fetchall()
    out = []
    for r in rows:
        try:
            vec = bytes_to_phases(r[5], dim=r[6] or store.embedding_dim)
        except Exception as e:
            logger.debug("decode hrr vector for memory %s failed: %s", r[0], e)
            continue
        out.append(
            {
                "id": r[0],
                "content": r[1],
                "memory_type": r[2],
                "created_at": r[3],
                "updated_at": r[4],
                "vector": vec,
            }
        )
    return out


def reason(
    store: HybridMemoryStore,
    entities: list[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """多实体组合检索（AND 语义）。

    对每个实体求其与记忆的相位相似度，取全部分量相似度，然后取最小值
    作为「该记忆同时涉及所有实体」的置信度（min = 严格 AND）。

    返回按 score 降序的记忆列表，带 per-entity 相似度明细。
    """
    if not entities:
        return []
    dim = store.embedding_dim
    entity_vecs = []
    for e in entities:
        try:
            entity_vecs.append((e, encode_text(e, dim)))
        except Exception as ex:
            logger.debug("encode entity %r failed: %s", e, ex)
    if not entity_vecs:
        return []

    memories = _all_hrr_memories(store)
    scored: list[tuple[float, dict[str, Any]]] = []
    for mem in memories:
        per_entity = {}
        sims = []
        for name, ev in entity_vecs:
            s = similarity(ev, mem["vector"])
            s01 = max(0.0, s)  # 0=无关，不产生负基线
            per_entity[name] = round(s01, 3)
            sims.append(s01)
        and_score = min(sims) if sims else 0.0
        mem["entity_sims"] = per_entity
        mem["score"] = round(and_score, 4)
        scored.append((and_score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]
    for _, mem in top:
        mem.pop("vector", None)
    return [mem for _, mem in top]


def contradict(
    store: HybridMemoryStore,
    threshold: float = 0.25,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """找可能互相矛盾的记忆对。

    定义：两条记忆的 HRR 相位相似度低于 threshold（内容完全不同），
    但检索行为却很接近 → 可能是同一主题下的不同说法。

    简化：直接取全库相位相似度最低的若干对，避免 O(n²) 全量计算。

    Returns:
        [{"a": {...}, "b": {...}, "similarity": float, "reason": str}, ...]
    """
    memories = _all_hrr_memories(store)
    n = len(memories)
    if n < 2:
        return []

    pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    # O(n²) 上限保护：超过 800 条只抽样最近 800 条
    if n > 800:
        memories.sort(key=lambda m: m.get("updated_at") or m.get("created_at") or "", reverse=True)
        memories = memories[:800]
        n = len(memories)

    for i in range(n):
        for j in range(i + 1, n):
            s = similarity(memories[i]["vector"], memories[j]["vector"])
            if s <= threshold:
                pairs.append((s, memories[i], memories[j]))

    pairs.sort(key=lambda t: t[0])  # 最不相似在前
    pairs = pairs[:limit]

    out = []
    for s, a, b in pairs:
        for mem in (a, b):
            mem.pop("vector", None)
            mem["score"] = None
        out.append(
            {
                "similarity": round(s, 4),
                "memory_a": a["id"],
                "memory_b": b["id"],
                "content_a": a["content"][:120],
                "content_b": b["content"][:120],
            }
        )
    return out
