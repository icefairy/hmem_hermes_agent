"""测试小 namespace 去重。"""
import logging, os, sys, signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, "/app")

from config import Settings
from engine.store import HybridMemoryStore
from engine.embeddings import EmbeddingClient
from engine.dedup import merge_similar

s = Settings()
db_root = s.db_root

ec = EmbeddingClient(
    base_url=s.embedding_base_url,
    api_key=s.embedding_api_key,
    embedding_model=s.embedding_model,
)

# 先测试 embedding 是否可用
print("Testing embedding...")
test_vec = ec.embed("hello test")
print(f"  embed ok, dim={len(test_vec)}")

# 选一个小 namespace
ns = "note"
db_path = os.path.join(db_root, f"{ns}.db")
store = HybridMemoryStore(db_path=db_path, embedding_dim=s.embedding_dim)
store.initialize()
try:
    print(f"\n=== {ns} ===")
    for mt in ["observation", "experience"]:
        before = store._count_by_type(mt)
        print(f"  {mt}: {before}")
        if before == 0:
            continue
        r = merge_similar(store, ec, mt, threshold=0.80)
        after = store._count_by_type(mt)
        print(f"  {mt} done: {before} -> {after} (removed {r['removed']})")
finally:
    store.close()
print("Done")