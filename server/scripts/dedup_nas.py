"""补全所有未嵌入的记忆 + merge_similar 去重。"""
import logging, os, sys, time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, "/app")

from config import Settings
from engine.store import HybridMemoryStore
from engine.embeddings import EmbeddingClient
from engine.dedup import merge_similar

s = Settings()
db_root = s.db_root
print(f"db_root: {db_root}")

ec = EmbeddingClient(
    base_url=s.embedding_base_url,
    api_key=s.embedding_api_key,
    embedding_model=s.embedding_model,
)

# Verify embedding works
print("Verifying embedding...")
test_vec = ec.embed("health check")
assert test_vec and len(test_vec) > 0, "Embedding failed!"
print(f"  OK, dim={len(test_vec)}")

namespaces = []
for fp in sorted(os.listdir(db_root)):
    if fp.endswith(".db"):
        ns = fp[:-3]
        namespaces.append(ns)

print(f"Namespaces: {namespaces}")

# Step 1: backfill embeddings for all unembedded items
print("\n=== Step 1: Backfill embeddings ===")
for ns in namespaces:
    db_path = os.path.join(db_root, f"{ns}.db")
    store = HybridMemoryStore(db_path=db_path, embedding_dim=s.embedding_dim)
    store.initialize()
    try:
        items = store.list_memories(limit=999999, offset=0)
        unembedded = [it for it in items if not it.get("embedded")]
        if not unembedded:
            print(f"  {ns}: all {len(items)} already embedded")
            continue
        print(f"  {ns}: {len(unembedded)}/{len(items)} need embedding...")
        ok = 0
        for it in unembedded:
            try:
                vec = ec.embed(it["content"])
                if vec:
                    # update content to mark it as embedded
                    store.update_memory(it["id"], it["content"], embedding=vec)
                    ok += 1
            except Exception as e:
                print(f"    id {it['id']} fail: {e}")
            time.sleep(0.05)  # rate limit
        print(f"    done: {ok}/{len(unembedded)} embedded")
    finally:
        store.close()

# Step 2: merge_similar for all namespaces
print("\n=== Step 2: merge_similar dedup ===")
total_removed = 0
for ns in namespaces:
    db_path = os.path.join(db_root, f"{ns}.db")
    store = HybridMemoryStore(db_path=db_path, embedding_dim=s.embedding_dim)
    store.initialize()
    try:
        print(f"\n  === {ns} ===")
        for mt in ["observation", "experience", "insight"]:
            before = store.count_by_type().get(mt, 0)
            if before == 0:
                print(f"    {mt}: 0 (skip)")
                continue
            try:
                r = merge_similar(store, ec, mt, threshold=0.80)
                after = store.count_by_type().get(mt, 0)
                rem = r.get("merged_count", r.get("removed", 0))
                total_removed += rem
                if rem > 0:
                    print(f"    {mt}: {before} -> {after} (removed {rem})")
                else:
                    print(f"    {mt}: {before} -> {after} (no change)")
            except Exception as e:
                print(f"    {mt}: ERROR {e}")
                import traceback; traceback.print_exc()
    finally:
        store.close()

print(f"\n=== Summary ===")
print(f"Total merged/deleted: {total_removed}")