"""End-to-end test: knowledge-base role for HMEM (Phase 2 / 方案 B).

Covers: v3->v4 migration on a legacy-style DB, knowledge-type write with doc fields,
document import/list/get/delete, time-decay exemption for knowledge, weighted
extra_namespaces merge, source provenance in search results.
"""
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time

import httpx
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    import main as server_main

    tmp = tempfile.mkdtemp(prefix="hmem-kb-test-")
    print(f"data dir: {tmp}")

    # ---- build a legacy v3-style db to exercise v3->v4 migration ----
    legacy_path = os.path.join(tmp, "legacy.db")
    conn = sqlite3.connect(legacy_path)
    conn.executescript(
        """
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            content_jieba TEXT NOT NULL DEFAULT '',
            memory_type TEXT NOT NULL DEFAULT 'experience',
            mem_action TEXT DEFAULT '',
            mem_context TEXT DEFAULT '{}',
            mem_outcome TEXT DEFAULT '{}',
            mem_metadata TEXT DEFAULT '{}',
            parent_id INTEGER DEFAULT NULL,
            hit_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT '2026-01-01 00:00:00',
            updated_at TEXT NOT NULL DEFAULT '2026-01-01 00:00:00'
        );
        """
    )
    conn.execute(
        "INSERT INTO memories(content, content_jieba, memory_type, created_at) "
        "VALUES ('旧经验 记录', '旧经验 记录', 'experience', '2026-08-01 10:00:00')"
    )
    conn.commit()
    conn.close()

    # launch server with tmp data dir
    os.environ["HMEM_DATA_DIR"] = tmp
    os.environ.setdefault("EMBEDDING_BASE_URL", "")
    os.environ.setdefault("EMBEDDING_API_KEY", "")
    os.environ.setdefault("HMEM_MIN_SCORE", "0")

    app = server_main.create_app()
    port = _free_port()
    BASE = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    # wait for startup: poll /health AND verify the documents route is live
    # (guards against a stale process answering on a recycled port)
    for _ in range(80):
        try:
            r = httpx.get(BASE + "/health", timeout=1)
            if r.status_code == 200:
                # route sanity: documents+stats must be mounted
                r2 = httpx.get(BASE + "/api/v1/stats", params={"namespace": "probe"}, timeout=1)
                if r2.status_code == 200:
                    break
        except Exception:
            time.sleep(0.1)
        else:
            time.sleep(0.1)
    # final readiness guard
    if httpx.get(BASE + "/health", timeout=1).status_code != 200:
        raise RuntimeError("server failed to start")

    c = httpx.Client(timeout=10, base_url=BASE)

    def post(path, **kw):
        r = c.post("/api/v1" + path, json=kw)
        return r.status_code, r.json()

    def get(path, **kw):
        r = c.get("/api/v1" + path, params=kw)
        return r.status_code, r.json()

    def delete(path, **kw):
        r = c.request("DELETE", "/api/v1" + path, params=kw)
        return r.status_code, r.json() if r.text else {}

    # ---- 1. legacy db migration ----
    print("\n[1] v3->v4 migration on legacy db")
    s, j = get("/stats", namespace="legacy")
    check("legacy stats returns 200", s == 200, str(j))
    check("legacy memory survived", j.get("total_memories") == 1, str(j))
    # verify doc columns now exist
    conn2 = sqlite3.connect(legacy_path)
    cols = [r[1] for r in conn2.execute("PRAGMA table_info(memories)").fetchall()]
    conn2.close()
    check("doc_id column added", "doc_id" in cols, str(cols))
    check("chunk_index column added", "chunk_index" in cols, str(cols))

    # ---- 2. knowledge type + doc fields on a fresh ns ----
    print("\n[2] knowledge-type write with doc fields (kb-eng ns)")
    ns = "kb-eng"
    for i, txt in enumerate(["部署手册 第一段 Kubernetes 安装", "部署手册 第二段 配置高可用"]):
        s, j = post(
            "/memories",
            content=txt,
            namespace=ns,
            memory_type="knowledge",
            mem_metadata={"kind": "knowledge_doc", "doc_id": "DEPLOY-1"},
        )
        check(f"write chunk {i} ok", s == 200 and j.get("embedded") is False, str(j))

    # ---- 3. document CRUD ----
    print("\n[3] document import / list / get / delete")
    content = (
        "知识库 文档 导入 测试。这段内容是关于持续集成流水线的完整说明，"
        "包含构建、测试、部署三个阶段。每一个阶段都有对应的脚本和参数配置。"
        "文档用于验证分块逻辑是否正确生效，以及检索时能否正确溯源到原始文档。"
    )
    s, j = post(
        "/documents",
        content=content,
        title="CI/CD 指南",
        uri="https://docs.example.com/ci-cd",
        doc_id="CI-CD",
        namespace=ns,
        chunk_size=30,
        overlap=5,
    )
    check("import doc 200", s == 200, str(j))
    chunks = j.get("chunks", 0)
    check("doc produced >1 chunk", chunks > 1, str(j))
    check("doc chunk count positive", chunks > 0, f"chunks={chunks}")

    s, j = get("/documents", namespace=ns)
    docids = [d["doc_id"] for d in j.get("documents", [])]
    check("list documents contains CI-CD", "CI-CD" in docids, str(docids))

    s, j = get(f"/documents/{'CI-CD'}", namespace=ns)
    check("get document detail 200", s == 200, str(j.get("doc_id")))
    check("get returns ordered chunks", all(
        j["chunks"][i]["chunk_index"] == i for i in range(len(j["chunks"]))
    ), f"n={len(j.get('chunks', []))}")

    # search hit should carry source
    s, j = post("/search", query="持续集成 流水线", namespace=ns)
    hit = next((r for r in j.get("results", []) if r.get("doc_id") == "CI-CD"), None)
    check("knowledge search returns CI-CD hit", hit is not None, str(j.get("results", [])[:1]))
    check("hit carries source provenance", bool(
        hit and hit.get("source", {}).get("uri") == "https://docs.example.com/ci-cd"
    ), str(hit))

    # ---- 4. time-decay exemption ----
    print("\n[4] time-decay exemption for knowledge")
    # store an old knowledge and an old ordinary memory, same relevance
    old_ts = "2026-01-01 00:00:00"
    s, j = post("/memories", content="Python异步编程", namespace=ns,
                memory_type="knowledge", created_at=old_ts)
    check("old knowledge write ok", s == 200, str(j))
    s, j = post("/memories", content="Python异步编程 个人学习笔记", namespace="pers",
                memory_type="experience", created_at=old_ts)
    check("old experience write ok", s == 200, str(j))
    s, j = post("/search", query="Python异步编程", namespace=ns)
    r = j.get("results", [])
    check("knowledge search finds old knowledge", any(x.get("memory_type") == "knowledge" for x in r), str(r[:1]))

    # ---- 5. weighted extra_namespaces merge ----
    print("\n[5] weighted extra_namespaces merge")
    # put a dedicated knowledge doc ONLY in the extra kb ns, search from app ns
    s, j = post("/documents", content="微服务架构 设计模式 分而治之",
                title="Arch", uri="http://a", doc_id="ARCH", namespace=ns)
    check("doc in kb ns ok", s == 200, str(j))
    s, j = post("/search", query="微服务 架构 设计模式", namespace="app",
                extra_namespaces=[ns])
    r = j.get("results", [])
    arch = next((x for x in r if x.get("doc_id") == "ARCH"), None)
    check("extra ns doc merged (kb-eng -> app search)", bool(arch), str(r[:1]))
    check("extra hit tagged _ns=kb-eng", bool(arch and arch.get("_ns") == ns), str(arch))
    check("extra hit carries source", bool(arch and arch.get("source", {}).get("uri") == "http://a"), str(arch))
    check("result carries _ns annotation", all("_ns" in x for x in r), str(r[:1]))
    # object-form (ns + weight) also accepted
    s, j2 = post("/search", query="微服务 架构 设计模式", namespace="app",
                 extra_namespaces=[{"ns": ns, "weight": 1.0}])
    check("object-form extra_namespaces supported", s == 200 and any(x.get("doc_id") == "ARCH" for x in j2.get("results", [])), str(j2))

    # ---- 6. delete document cascades ----
    print("\n[6] delete document cascades")
    s, j = delete(f"/documents/{'CI-CD'}", namespace=ns)
    check("delete doc 200", s == 200 and j.get("deleted") is True, str(j))
    s, j = get(f"/documents/{'CI-CD'}", namespace=ns)
    check("doc gone after delete", s == 404, str(j))
    s, j = post("/search", query="持续集成 流水线", namespace=ns)
    check("no CI-CD hit after delete", all(x.get("doc_id") != "CI-CD" for x in j.get("results", [])), str(j.get("results", [])[:1]))

    # ---- 7. stats document_count ----
    print("\n[7] stats includes document_count")
    s, j = get("/stats", namespace=ns)
    check("stats has document_count", "document_count" in j, str(j))
    check("document_count is int", isinstance(j.get("document_count"), int), str(j))

    srv.should_exit = True
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n===== RESULT: {PASS} passed, {FAIL} failed =====")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
