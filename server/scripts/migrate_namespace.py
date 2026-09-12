#!/usr/bin/env python3
"""HMEM 命名空间迁移 —— 幂等增量版。

用法：
    # 预览（默认，不改数据）
    python migrate_namespace.py --src novel --dst default
    # 真正执行
    python migrate_namespace.py --src novel --dst default --apply
    # 只导指定源 id（精确定向补迁）
    python migrate_namespace.py --src novel --dst default --only 48,835,3450 --apply

设计要点（均为实战教训，勿改）：
  1. **幂等**：按「内容精确比对」判断是否已存在，已存在则跳过。
     绝不假设"源里所有条目都要导"——重跑安全是第一要求。
     2026-09-12 事故：全量重跑脚本二次执行造成 98 条冗余。
  2. **保留原字段**：memory_type / mem_action / mem_context / mem_outcome /
     mem_metadata / created_at / hit_count / importance / parent_id。
     POST /memories 不支持 hit_count/importance/parent_id，故阶段 B 用服务端
     连接（含 sqlite-vec）补写。
  3. **先备份**：调用 POST /backup，并额外落一份 .db 快照。
  4. **parent_id 两遍**：先建全部条目，再按 id 映射回填父子关系。
  5. **禁止直接 sqlite3 操作 vec_memories**：裸 sqlite3 无 vec0 扩展会抛
     `no such module: vec0`，导致所在事务整体回滚（删除静默失效）。
     一律用 HybridMemoryStore（服务端连接）。
  6. **内容变更需重建 FTS**：INSERT INTO memories_fts(memories_fts) VALUES('rebuild')。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("HMEM_API_URL", "http://localhost:8090") + "/api/v1"
KEY = os.environ.get("HMEM_API_KEY", "")
DB_ROOT = os.environ.get("HMEM_DATA_DIR", "/data/hmem-data")
SERVER_DIR = os.environ.get("HMEM_SERVER_DIR", "/data/codes/hmem/server")

# store._heuristic_importance 基准值：与这些不同的 importance 需要显式复原
HEURISTIC = {
    "self_identity": 0.95, "mental_model": 0.85, "anchor": 0.85,
    "insight": 0.75, "knowledge": 0.6, "experience": 0.5, "observation": 0.4,
}


def api(method: str, path: str, body=None, timeout: int = 90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read()[:200]!r}") from e


def snapshot(ns: str) -> str:
    """落一份 .db 物理快照（比 gz 备份更快可回滚）。"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = f"{DB_ROOT}/backups/{ns}_premigrate_{ts}.db"
    shutil.copy(f"{DB_ROOT}/{ns}.db", dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="default")
    ap.add_argument("--only", default="", help="逗号分隔的源 id，仅迁移这些")
    ap.add_argument("--apply", action="store_true", help="不加则只预览")
    ap.add_argument("--skip-backup", action="store_true")
    args = ap.parse_args()

    src_db, dst_db = f"{DB_ROOT}/{args.src}.db", f"{DB_ROOT}/{args.dst}.db"
    for p in (src_db, dst_db):
        if not os.path.isfile(p):
            print(f"[!] 库不存在: {p}")
            return 1

    src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True, timeout=30)
    where, params = "", []
    if args.only:
        ids = [int(x) for x in args.only.split(",") if x.strip()]
        where = f"WHERE id IN ({','.join('?' * len(ids))})"
        params = ids
    rows = src.execute(
        f"SELECT id, content, memory_type, mem_action, mem_context, mem_outcome, "
        f"       mem_metadata, parent_id, hit_count, importance, created_at, pinned "
        f"FROM memories {where} ORDER BY id", params
    ).fetchall()
    print(f"[+] 源 {args.src}: 读取 {len(rows)} 条")

    # ── 全库内容集合：幂等判据 ──
    # 两种「已存在」：① 内容精确匹配；② 内容已作为片段并入某条（去重合并的
    # `---` 分段）。② 也算已存在——信息已可检索，重复导入只会制造噪音。
    dst = sqlite3.connect(f"file:{dst_db}?mode=ro", uri=True, timeout=30)
    all_content = [r[0] for r in dst.execute("SELECT content FROM memories")]
    dst.close()
    existing = set(all_content)
    blob = "\n".join(all_content)
    FRAG = 60  # 片段匹配前缀长度

    def is_present(content: str) -> tuple[bool, str]:
        if content in existing:
            return True, "精确"
        if len(content) >= FRAG and content[:FRAG] in blob:
            return True, "片段"
        return False, ""

    todo, present_exact, present_frag = [], 0, 0
    for r in rows:
        ok, how = is_present(r[1])
        if not ok:
            todo.append(r)
        elif how == "精确":
            present_exact += 1
        else:
            present_frag += 1
    print(f"[+] 已存在跳过: 精确 {present_exact} / 已并入他条 {present_frag}"
          f" / 待导入 {len(todo)}")
    if not todo:
        print("[+] 无差异，无需迁移（幂等 ✅）")
        src.close()
        return 0
    for r in todo:
        print(f"    + #{r[0]} {r[2]:12s} len={len(r[1]):5d} | {r[1][:60]!r}".replace("\\n", " "))

    if not args.apply:
        print("\n[预览模式] 加 --apply 执行")
        src.close()
        return 0

    if not args.skip_backup:
        print("[+] 备份:", snapshot(args.dst))
        try:
            res = api("POST", "/backup")
            print("    gz 备份:", ", ".join(
                b["filename"] for b in res["backups"] if b["namespace"] == args.dst))
        except Exception as e:  # noqa: BLE001
            print("    gz 备份失败(不阻断):", str(e)[:120])

    # ── 阶段 A：经 API 导入 ──
    t0, id_map, children, meta_fix, failed = time.time(), {}, [], [], []
    for i, r in enumerate(todo, 1):
        (oid, content, mt, action, ctx, outcome, meta, parent_id,
         hit, imp, created, pinned) = r
        try:
            res = api("POST", "/memories", {
                "content": content, "namespace": args.dst, "memory_type": mt,
                "mem_action": action or "",
                "mem_context": json.loads(ctx or "{}"),
                "mem_outcome": json.loads(outcome or "{}"),
                "mem_metadata": json.loads(meta or "{}"),
                "created_at": created,
                "pinned": bool(pinned),   # 保护标记随迁，防去重吞噬
            })
        except Exception as e:  # noqa: BLE001
            failed.append((oid, str(e)[:200]))
            print(f"  [!] {oid} 失败: {str(e)[:200]}")
            continue
        nid = res["memory_id"]
        id_map[oid] = nid
        if parent_id:
            children.append((nid, parent_id, oid))
        if (hit or 0) > 0 or abs((imp or 0) - HEURISTIC.get(mt, 0.5)) > 1e-6:
            meta_fix.append((nid, hit or 0, imp))
        if i % 20 == 0:
            print(f"  ... {i}/{len(todo)}")
    print(f"[+] 阶段A: 成功 {len(id_map)} / 失败 {len(failed)}，{time.time()-t0:.1f}s")
    if failed:
        print("[!] 有失败条目，跳过阶段B")
        return 2

    # ── 阶段 B：补 parent_id / hit_count / importance（需服务端连接）──
    sys.path.insert(0, SERVER_DIR)
    from engine.store import HybridMemoryStore  # noqa: E402

    s = HybridMemoryStore(db_path=dst_db, embedding_dim=1024)
    s.initialize()
    c = s._conn
    c.execute("PRAGMA busy_timeout=60000")

    # parent_id 两遍：children 后建，父级已在 id_map 中
    fixed_p = 0
    for nid, old_parent, oid in children:
        new_parent = id_map.get(old_parent)
        if not new_parent:
            # 父级本身是已存在条目 → 在 dst 中按内容反查
            prow = next((x for x in rows if x[0] == old_parent), None)
            if prow:
                found = c.execute("SELECT id FROM memories WHERE content = ?",
                                  (prow[1],)).fetchone()
                new_parent = found[0] if found else None
        if not new_parent:
            print(f"  [!] #{oid} 父级 {old_parent} 无法解析，跳过")
            continue
        c.execute("UPDATE memories SET parent_id=? WHERE id=?", (new_parent, nid))
        fixed_p += 1
    for nid, hit, imp in meta_fix:
        c.execute("UPDATE memories SET hit_count=?, importance=? WHERE id=?",
                  (hit, imp, nid))
    c.commit()
    if fixed_p or meta_fix:
        c.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        c.commit()
    print(f"[+] 阶段B: parent_id {fixed_p} 条，hit/importance {len(meta_fix)} 条")

    total = s.count_memories()
    print(f"[+] 目标 {args.dst} 现有 {total} 条")
    s.close()
    src.close()

    # ── 校验：源内容是否全部可达 ──
    d2 = sqlite3.connect(f"file:{dst_db}?mode=ro", uri=True, timeout=30)
    dset = {r[0] for r in d2.execute("SELECT content FROM memories")}
    blob = "\n".join(r[0] for r in d2.execute("SELECT content FROM memories"))
    exact = sum(1 for r in todo if r[1] in dset)
    merged = sum(1 for r in todo if r[1] not in dset and r[1][:60] in blob)
    lost = [r[0] for r in todo if r[1] not in dset and r[1][:60] not in blob]
    print(f"[+] 校验: 精确在库 {exact} / 已并入他条 {merged} / 丢失 {len(lost)}")
    if lost:
        print("    [!] 丢失源 id:", lost)
    d2.close()
    return 0 if not lost else 3


if __name__ == "__main__":
    raise SystemExit(main())
