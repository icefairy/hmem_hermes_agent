#!/usr/bin/env python3
"""把 HMEM 里 v7 分片上线前的历史超长记忆补做分片。

背景：store.py 的 add_memory_sharded（v7）会把超长内容切成 <=3000 字符的分片，
父行保留完整原文但不参与检索，分片才是可检索单元。检索侧 _aggregate_shards
把命中分片折叠回父行。但**分片上线前写入的存量记忆** shard_total=1、无分片行，
仍以整篇参与 rerank → 上游 3000 字符截断 → 一次搜索几十次串行请求 → 事件循环卡死。

本脚本对存量超长行补做分片，逻辑与 add_memory_sharded 一致。

用法:
    python3 shard_backfill.py --db <path> [--dry-run] [--min-chars 3000]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/codes/hmem/server")

from engine.store import (  # noqa: E402
    HybridMemoryStore,
    _split_content,
    _SHARD_CHARS,
)

CONF = "/etc/supervisor/conf.d/hmem.conf"


def read_conf_key() -> tuple[str, str, str]:
    """从 supervisor 配置里读嵌入端点与 key（不硬编码凭据）。"""
    base, key, model = "http://localhost:3001/v1", "", "bge-m3"
    txt = Path(CONF).read_text()
    for pat, name in (
        (r'EMBEDDING_BASE_URL="([^"]+)"', "base"),
        (r'EMBEDDING_API_KEY="([^"]+)"', "key"),
        (r'EMBEDDING_MODEL="([^"]+)"', "model"),
    ):
        m = re.search(pat, txt)
        if m:
            if name == "base":
                base = m.group(1)
            elif name == "key":
                key = m.group(1)
            else:
                model = m.group(1)
    return base, key, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-chars", type=int, default=_SHARD_CHARS)
    args = ap.parse_args()

    db = Path(args.db)
    if not db.is_file():
        print(f"!! 找不到 {db}")
        return 1

    base, key, model = read_conf_key()
    client = None
    if key and not args.dry_run:
        from engine.embeddings import EmbeddingClient

        client = EmbeddingClient(
            base_url=base, api_key=key, embedding_model=model, embedding_dim=1024
        )

    store = HybridMemoryStore(db_path=str(db), embedding_dim=1024)
    store.initialize()

    rows = store._conn.execute(
        "SELECT id, content, memory_type, importance, doc_id, doc_uri, doc_title, "
        "       doc_category, doc_tags, created_at, mem_action, mem_context, mem_outcome, mem_metadata "
        "FROM memories "
        "WHERE shard_of IS NULL AND (shard_total IS NULL OR shard_total <= 1) "
        "  AND length(content) > ? "
        "ORDER BY length(content) DESC",
        (args.min_chars,),
    ).fetchall()

    print(f"== {db.name}: 待分片 {len(rows)} 条")
    total_shards = 0
    done = 0
    failed: list[tuple[int, str]] = []

    for r in rows:
        (
            pid, content, mtype, importance, doc_id, doc_uri, doc_title,
            doc_category, doc_tags, created_at, mem_action, mem_context,
            mem_outcome, mem_metadata,
        ) = r
        pieces = _split_content(content)
        if len(pieces) <= 1:
            continue

        if args.dry_run:
            print(f"  [dry] id={pid} {len(content)}字符 → {len(pieces)}片")
            total_shards += len(pieces)
            done += 1
            continue

        # 逐片写入（复用 store 的 add_memory：自动处理 FTS 触发器 / HRR / 向量）
        written = 0
        for i, piece in enumerate(pieces):
            vec = None
            try:
                vec = client.embed(piece) if client else None
            except Exception as e:  # noqa: BLE001
                print(f"    片 {i+1} 向量化失败: {e}")
            sid = store.add_memory(
                content=piece,
                embedding=vec,
                memory_type=mtype or "experience",
                mem_action=mem_action,
                mem_context=mem_context,
                mem_outcome=mem_outcome,
                mem_metadata=mem_metadata,
                importance=importance,
                doc_id=doc_id or None,
                doc_uri=doc_uri or None,
                doc_title=doc_title or None,
                doc_category=doc_category or None,
                doc_tags=doc_tags or None,
                created_at=created_at,
                shard_of=pid,
                shard_total=0,
            )
            if sid:
                written += 1
            else:
                print(f"    片 {i+1}/{len(pieces)} 写入失败 (parent={pid})")

        if written == 0:
            failed.append((pid, "所有分片写入失败"))
            continue

        # 分片就绪后，把父行标记为非检索（保留完整原文，供聚合回填）
        store._conn.execute(
            "UPDATE memories SET shard_total=? WHERE id=?", (written, pid)
        )
        # 父行不再参与向量检索，清掉它的向量避免留下陈旧数据
        store._conn.execute("DELETE FROM vec_memories WHERE memory_id=?", (pid,))
        store._conn.commit()

        total_shards += written
        done += 1
        if done % 10 == 0:
            print(f"  ...已处理 {done}/{len(rows)}")

    store.close()

    print(f"== 完成: {done} 条父记忆 → {total_shards} 片")
    if failed:
        print(f"!! {len(failed)} 条失败: {failed[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
