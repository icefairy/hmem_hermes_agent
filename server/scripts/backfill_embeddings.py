"""为存量记忆批量回填 bge-m3 embedding（写入 vec_memories 表）。

用法（在 server 目录下，用 hmem venv python 运行）:
    python scripts/backfill_embeddings.py piagent          # 只回填指定 namespace（可多个，空格分隔）
    python scripts/backfill_embeddings.py --all            # 回填 db_root 下所有 namespace
无参数时默认对所有 namespace 运行；已嵌入（embedded=True）的记忆自动跳过。

依赖环境变量（与 supervisor hmem.conf 保持一致）:
    EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL
    EMBEDDING_DIM（默认 1024）/ HMEM_DATA_DIR
"""
import logging
import os
import sqlite3
import sys
import time

import sqlite_vec

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
# 降噪：不打印每个 HTTP 请求的成功日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from engine.store import HybridMemoryStore
from engine.embeddings import EmbeddingClient


def _embedded_ids(db_path: str) -> set[int]:
    """已存在于 vec_memories 表中的 memory_id 集合（幂等判断）。"""
    conn = sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return {r[0] for r in conn.execute("SELECT memory_id FROM vec_memories")}
    finally:
        conn.close()

s = Settings()
db_root = s.db_root

if not (s.embedding_base_url and s.embedding_api_key):
    print("!! EMBEDDING_BASE_URL / EMBEDDING_API_KEY 未设置，无法回填", file=sys.stderr)
    sys.exit(1)

ec = EmbeddingClient(
    base_url=s.embedding_base_url,
    api_key=s.embedding_api_key,
    embedding_model=s.embedding_model,
    rerank_model=s.rerank_model,
    embedding_dim=s.embedding_dim,
)

args = sys.argv[1:]
if not args or "--all" in args:
    namespaces = sorted(
        f[:-3] for f in os.listdir(db_root) if f.endswith(".db")
    )
else:
    namespaces = args

print(f"db_root      : {db_root}")
print(f"embedding url: {s.embedding_base_url}")
print(f"model/dim    : {s.embedding_model} / {s.embedding_dim}")
print(f"namespaces   : {namespaces}")

# 先做一次健康验证
test_vec = ec.embed("health check")
if not test_vec:
    print("ERROR: embedding 健康检查失败，中止", file=sys.stderr)
    sys.exit(1)
print(f"embedding OK, dim={len(test_vec)}")

total_ok = total_need = 0
for ns in namespaces:
    db_path = os.path.join(db_root, f"{ns}.db")
    if not os.path.isfile(db_path):
        print(f"  [{ns}] 跳过：无数据库文件 {db_path}")
        continue
    store = HybridMemoryStore(db_path=db_path, embedding_dim=s.embedding_dim)
    store.initialize()
    try:
        items = store.list_memories(limit=999999, offset=0)
        embedded = _embedded_ids(db_path)
        unembedded = [it for it in items if it["id"] not in embedded]
        if not unembedded:
            print(f"  [{ns}] 共 {len(items)} 条，全部已嵌入，跳过")
            continue
        ok = 0
        for it in unembedded:
            try:
                vec = ec.embed(it["content"])
                if vec:
                    store.update_memory(it["id"], it["content"], embedding=vec)
                    ok += 1
            except Exception as e:  # noqa: BLE001 — 单条失败不中断整体
                print(f"    id {it['id']} 失败: {e}")
            time.sleep(0.05)  # 限速，避免打爆网关
        total_ok += ok
        total_need += len(unembedded)
        print(f"  [{ns}] 回填 {ok}/{len(unembedded)}")
    finally:
        store.close()

print(f"\n总计：回填 {total_ok}/{total_need} 条记忆的 embedding")