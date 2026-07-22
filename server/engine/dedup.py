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


def merge_similar(
    store: HybridMemoryStore,
    embedding_client: EmbeddingClient | None,
    memory_type: str,
    threshold: float = _SIM_THRESHOLD,
    batch_size: int = 50,
) -> dict[str, Any]:
    """Scan all items of a given type, cluster semantically similar ones,
    merge each cluster into a single consolidated entry.

    Returns stats: {merged_count, kept_count, deleted_ids, errors}
    """
    items = store.list_memories(memory_type=memory_type, limit=999999, offset=0)
    if not items:
        return {"merged_count": 0, "kept_count": len(items), "errors": []}

    logger.info("merge_similar(%s): scanning %d items", memory_type, len(items))

    # Get embeddings for all items (batch embed)
    contents = [it["content"] for it in items]
    embeddings: list[list[float] | None] = []
    if embedding_client and contents:
        try:
            embeddings = embedding_client.embed_batch(contents)
            if embeddings is None:
                embeddings = []
        except Exception as e:
            logger.warning("batch embed failed: %s", e)
            embeddings = []

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

        if best_cluster and best_sim >= threshold:
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
    deleted_ids: list[int] = []

    for cl in clusters:
        if len(cl["ids"]) <= 1:
            kept_count += 1
            continue

        # Merge: keep the longest / most detailed content as the survivor
        ids = cl["ids"]
        contents = cl["contents"]

        # Pick the longest content as the master
        master_idx = max(range(len(contents)), key=lambda i: len(contents[i] or ""))
        master_id = ids[master_idx]
        master_content = contents[master_idx]

        # Delete all other items
        for i, item_id in enumerate(ids):
            if i == master_idx:
                continue
            try:
                store.delete_memory(item_id)
                deleted_ids.append(item_id)
            except Exception as e:
                logger.warning("  delete %d failed: %s", item_id, e)

        # Update master content to be a consolidated version
        if len(contents) > 1:
            # Optionally flag the merged entry
            consolidated = master_content
            try:
                store.update_memory(master_id, consolidated)
            except Exception as e:
                logger.warning("  update %d failed: %s", master_id, e)

        merged_count += len(ids) - 1
        kept_count += 1

    logger.info(
        "  result: %d merged into %d kept (%d ids deleted)",
        merged_count,
        kept_count,
        len(deleted_ids),
    )
    return {
        "merged_count": merged_count,
        "kept_count": kept_count,
        "deleted_ids": deleted_ids,
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