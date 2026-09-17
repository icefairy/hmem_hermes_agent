"""Deduplication utilities for Reflect Engine.

Two modes:
  1. merge_similar — batch merge existing items of a given memory_type
  2. dedup_candidates — inline dedup before adding new items (used by reflect pipeline)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from engine.store import HybridMemoryStore
from engine.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

# Similarity threshold (cosine). Higher = stricter.
_SIM_THRESHOLD = 0.85

# 长文专用阈值：单条 >= _LONG_NOTE_CHARS 字时，阈值升到 _LONG_SIM_THRESHOLD。
# 长文档（课程/经典精读笔记）的嵌入天然趋同——同为「卜筮/易经」主题的不同讲次
# 很容易越过 0.80/0.85 被判为重复。2026-09-12 事故中《卜筮正术》各讲正是因此
# 成批互判重复。长文一旦误合并，损失的是整篇不可重建的内容，故取严。
_LONG_NOTE_CHARS = 500
_LONG_SIM_THRESHOLD = 0.92

# 合并后总长上限（2026-09-17 修复风雪球）。
# 被并入的内容（主条 + 各小节 + "\n\n---\n\n" 分隔符）合计不得超此值。
# 与 store._MAX_CONTENT_CHARS(8000) 保持同量级：去重的目的是减信息，
# 合并结果不应超过单条写入上限，否则又会被 store 层截断（信息白归档）。
_MAX_MERGED_CHARS = 8000


def merge_similar(
    store: HybridMemoryStore,
    embedding_client: EmbeddingClient | None,
    memory_type: str,
    threshold: float = _SIM_THRESHOLD,
    batch_size: int = 50,
) -> dict[str, Any]:
    """Scan all items of a given type, cluster semantically similar ones,
    merge each cluster into a single consolidated entry.

    Returns stats: {merged_count, kept_count, superseded_ids, deleted_ids, errors}

    注意：本函数**永不硬删除**。被合并的条目一律归档（archive_memory，可逆），
    且所有成员内容都会汇总进保留条目。deleted_ids 恒为空数组（历史兼容）。
    另：pinned / no_forget 记忆一律跳过去重，绝不参与合并。
    """
    items = store.list_memories(memory_type=memory_type, limit=999999, offset=0)
    if not items:
        return {"merged_count": 0, "kept_count": len(items), "errors": []}

    # ── 保护：pinned / no_forget 记忆永不参与合并删除 ──
    # 2026-09-12 事故：课程精读笔记（同为 insight、术语高度重叠）被判重后
    # 硬删除且不合并内容，导致整篇笔记永久丢失。pinned 是用户的「永久保留」
    # 显式意图，任何自动去重都必须让路。
    pinned_items = [it for it in items if it.get("pinned") or it.get("no_forget")]
    if pinned_items:
        pinned_ids = {it["id"] for it in pinned_items}
        items = [it for it in items if it["id"] not in pinned_ids]
        logger.info(
            "merge_similar(%s): %d pinned/no_forget 条已跳过去重（受保护）",
            memory_type,
            len(pinned_items),
        )
    _keep = len(pinned_items)

    logger.info("merge_similar(%s): scanning %d items", memory_type, len(items))

    # Get embeddings for all items — 库里已存的向量直接读（写入路径已 embed 过），
    # 只对缺向量的条目调 embed API 补齐并回写，避免每轮全量重复调 API（深夜空转的根源）
    contents = [it["content"] for it in items]
    vec_map: dict[int, list[float]] = {}
    try:
        vec_map = store.list_vectors(memory_type=memory_type)
    except Exception as e:
        logger.warning("list_vectors failed: %s", e)
    embeddings: list[list[float] | None] = [vec_map.get(it["id"]) for it in items]

    if embedding_client and contents:
        missing = [i for i, e in enumerate(embeddings) if e is None]
        if missing:
            try:
                new_vecs = embedding_client.embed_batch(
                    [contents[i] for i in missing]
                )
                if new_vecs:
                    for i, vec in zip(missing, new_vecs):
                        if vec is not None:
                            embeddings[i] = vec
                            try:
                                store.upsert_vector(items[i]["id"], vec)
                            except Exception:
                                pass  # 回写失败不影响本轮去重
            except Exception as e:
                logger.warning("batch embed (missing only, n=%d) failed: %s", len(missing), e)

    # Pad to same length
    while len(embeddings) < len(contents):
        embeddings.append(None)

    # Greedy clustering: for each item, check against cluster centroids
    clusters: list[dict[str, Any]] = []  # each: {centroid, ids, contents}
    skipped_no_embed = 0
    for idx, item in enumerate(items):
        emb = embeddings[idx]
        if emb is None:
            skipped_no_embed += 1
            continue

        best_cluster = None
        best_sim = 0.0
        for cl in clusters:
            sim = _cosine_similarity(emb, cl["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_cluster = cl

        # 长文保护：本条或候选簇中出现长文，阈值升到 _LONG_SIM_THRESHOLD。
        # 任一方是长文即从严——防长文被短条目「拉进」簇里连带归档。
        eff_threshold = threshold
        if len(item["content"] or "") >= _LONG_NOTE_CHARS or any(
            len(c or "") >= _LONG_NOTE_CHARS for c in (best_cluster or {}).get("contents", [])
        ):
            eff_threshold = max(threshold, _LONG_SIM_THRESHOLD)

        if best_cluster and best_sim >= eff_threshold:
            best_cluster["ids"].append(item["id"])
            best_cluster["contents"].append(item["content"])
            # Update centroid as running average
            n = len(best_cluster["ids"])
            c = best_cluster["centroid"]
            for i in range(len(c)):
                c[i] = c[i] + (emb[i] - c[i]) / n
        else:
            clusters.append({
                "centroid": emb[:],
                "ids": [item["id"]],
                "contents": [item["content"]],
            })

    if skipped_no_embed:
        logger.info("  %d items skipped (no embedding)", skipped_no_embed)

    # Clusters with only 1 item are kept as-is
    merged_count = 0
    kept_count = 0
    deleted_ids: list[int] = []  # 恒为空：去重永不硬删除（保留键，兼容旧调用方）
    superseded_ids: list[int] = []  # 被并入保留条目的 id（已归档，可恢复）

    for cl in clusters:
        if len(cl["ids"]) <= 1:
            kept_count += 1
            continue

        # ── 合并：内容「相加」而非「取最长」 ──
        # 2026-09-12 事故：原实现只保留最长的一条内容、其余条目连内容一起硬删除，
        # 导致课程精读笔记整篇永久丢失（《卜筮正术》第003讲 #12175）。
        # 现在：被合并条目的内容全部保留在库中（可逆归档），并将所有人内容
        # 汇总进「保留条目」，确保任何一条的独有信息都不会蒸发。
        ids = cl["ids"]
        contents = cl["contents"]

        # 保留最长的一条作为主条目（信息量最大）
        master_idx = max(range(len(contents)), key=lambda i: len(contents[i] or ""))
        master_id = ids[master_idx]
        master_content = contents[master_idx]

        # 汇总合并：主条 + 其余条各自作为独立小节保留，不丢任何内容
        #
        # 总长上限（2026-09-17 修复雪球）：原实现无条件追加所有入 cluster 的内容，
        # 而合并后的主条下次可能再被并入另一个 cluster → 雪球式增长
        # （实测单条达 29 万字符 = 1920 段拼接，含 99% 重复）。
        # 去重的目的本是**减**信息，不该越合越大。现在设预算，超预算停止追加。
        # 被跳过条目的内容仍在库中（已归档、可恢复），不是丢失。
        others = [contents[i] for i in range(len(contents)) if i != master_idx]
        if others:
            budget = _MAX_MERGED_CHARS - len(master_content)
            accepted: list[str] = []
            skipped = 0
            for c in others:
                if not c:
                    continue
                # +6 是 "\n\n---\n\n" 分隔符的开销
                if len(c) + 6 > budget:
                    skipped += 1
                    continue
                accepted.append(c)
                budget -= len(c) + 6
            if accepted:
                merged_text = (
                    master_content + "\n\n---\n\n" + "\n\n---\n\n".join(accepted)
                )
            else:
                merged_text = master_content
            if skipped:
                logger.info(
                    "  merge %d: 已拼 %d 段，跳过 %d 段（达 %d 字符预算；跳过内容已归档可恢复）",
                    master_id, len(accepted), skipped, _MAX_MERGED_CHARS,
                )
        else:
            merged_text = master_content

        # 归档（不删除）其余条目——内容与向量都还在，可随时恢复
        # 注意：archive_memory 拒绝时【返回 False 而不抛异常】（store 层保护），
        # 因此必须检查返回值，否则会把未归档的 id 误记入 superseded_ids。
        for i, item_id in enumerate(ids):
            if i == master_idx:
                continue
            try:
                if store.archive_memory(item_id):
                    superseded_ids.append(item_id)
                else:
                    logger.warning(
                        "  archive %d refused（受保护或不存在）—— 未并入", item_id
                    )
            except Exception as e:
                logger.warning("  archive %d failed: %s", item_id, e)

        try:
            store.update_memory(master_id, merged_text)
        except Exception as e:
            logger.warning("  update %d failed: %s", master_id, e)

        merged_count += len(ids) - 1
        kept_count += 1

    logger.info(
        "  result: %d merged into %d kept (%d ids archived)",
        merged_count,
        kept_count,
        len(superseded_ids),
    )
    return {
        "merged_count": merged_count,
        "kept_count": kept_count + _keep,
        "deleted_ids": deleted_ids,
        "superseded_ids": superseded_ids,
        "errors": [],
    }


def dedup_before_add(
    store: HybridMemoryStore,
    embedding_client: EmbeddingClient | None,
    candidate_content: str,
    target_type: str,
    threshold: float = _SIM_THRESHOLD,
) -> int | None:
    """Check if a semantically similar item of `target_type` already exists.

    Returns the existing memory ID if found, None if no match.
    """
    if not embedding_client:
        return None

    # Quick FTS5 first — if exact or near-exact match, skip
    fts_results = store.search_fts(candidate_content, limit=3)
    for r in fts_results:
        if r.get("memory_type") != target_type:
            continue
        existing = r.get("content", "")
        # Very similar — rough char overlap check
        overlap = _char_overlap(candidate_content, existing)
        if overlap > 0.92:
            logger.debug(
                "dedup: FTS match (%.2f) for '%s…' → use existing %d",
                overlap,
                candidate_content[:60],
                r["id"],
            )
            return r["id"]

    # If FTS didn't match, try vector search
    query_emb = embedding_client.embed(candidate_content)
    if not query_emb:
        return None

    vec_results = store.search_vector(query_emb, limit=5)
    for r in vec_results:
        if r.get("memory_type") != target_type:
            continue
        sim = r.get("score", 0.0) or r.get("vec_similarity", 0.0)
        if sim >= threshold:
            logger.debug(
                "dedup: vector match (%.3f) for '%s…' → use existing %d",
                sim,
                candidate_content[:60],
                r["id"],
            )
            return r["id"]

    return None


# -- Helpers -----------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(av * bv for av, bv in zip(a, b))
    na = sum(av * av for av in a) ** 0.5
    nb = sum(bv * bv for bv in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _char_overlap(a: str, b: str) -> float:
    """Character-level overlap ratio (for quick dedup check)."""
    if not a or not b:
        return 0.0
    a_set = set(a)
    b_set = set(b)
    inter = a_set & b_set
    return len(inter) / max(len(a_set), len(b_set))