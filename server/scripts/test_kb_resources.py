"""知识库资源接口测试 — /api/v1/knowledge*/knowledge-bases* 增删查。

验证:
  1. POST /knowledge            新增单条知识（含 category/tags/doc_id）
  2. GET  /knowledge            列表 + 按 category/tags/doc_id 过滤
  3. GET  /knowledge/{id}       单条详情
  4. DELETE /knowledge/{id}     删除单条
  5. GET  /knowledge/categories 分类汇总
  6. POST/GET/DELETE /knowledge-bases 库级管理
"""
import os
import shutil
import socket
import sys
import tempfile
import threading
import time

import httpx
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    import main as server_main

    tmp = tempfile.mkdtemp(prefix="hmem-kb-res-")
    os.environ["HMEM_DATA_DIR"] = tmp
    os.environ.setdefault("EMBEDDING_BASE_URL", "")
    os.environ.setdefault("EMBEDDING_API_KEY", "")
    os.environ.setdefault("HMEM_MIN_SCORE", "0")

    app = server_main.create_app()
    port = free_port()
    BASE = f"http://127.0.0.1:{port}"
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(80):
        try:
            if httpx.get(BASE + "/health", timeout=1).status_code == 200 and httpx.get(
                BASE + "/api/v1/stats", params={"namespace": "p"}, timeout=1
            ).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
        else:
            time.sleep(0.1)
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

    ns = "kb-res"

    print("\n[1] create knowledge-base + single knowledge entries")
    s, j = post("/knowledge-bases", namespace=ns, title="研发文档库")
    check("create kb 200", s == 200 and j.get("namespace") == ns, str(j))

    s, j = post("/knowledge", namespace=ns,
                content="K8s 滚动更新策略配置", category="ops/deploy", tags=["k8s", "部署"])
    check("add knowledge 1", s == 200 and j.get("memory_id") is not None, str(j))

    s, j = post("/knowledge", namespace=ns,
                content="FastAPI 依赖注入最佳实践", category="dev/backend", tags=["fastapi", "python"])
    check("add knowledge 2", s == 200, str(j))
    kid2 = j.get("memory_id")

    s, j = post("/knowledge", namespace=ns,
                content="Nginx 反向代理缓存配置", category="ops/proxy", tags=["nginx"], doc_id="NGX-1")
    check("add knowledge 3 (with doc_id)", s == 200, str(j))
    kid3 = j.get("memory_id")

    print("\n[2] list + category/tags/doc_id filtering")
    s, j = get("/knowledge", namespace=ns)
    ks = j.get("results", [])
    check("list returns all 3", s == 200 and len(ks) == 3, f"n={len(ks)}")
    check("entries are knowledge type", all(x.get("memory_type") == "knowledge" for x in ks), str(ks[:1]))

    s, j = get("/knowledge", namespace=ns, category="ops/deploy")
    ks = j.get("results", [])
    check("filter by category ops/deploy", s == 200 and len(ks) == 1 and ks[0]["content"].startswith("K8s"), str(ks))

    s, j = get("/knowledge", namespace=ns, tags="k8s")
    ks = j.get("results", [])
    check("filter by tag k8s", s == 200 and len(ks) == 1, str(ks))

    s, j = get("/knowledge", namespace=ns, doc_id="NGX-1")
    ks = j.get("results", [])
    check("filter by doc_id NGX-1", s == 200 and len(ks) == 1 and ks[0]["content"].startswith("Nginx"), str(ks))

    print("\n[3] get single detail")
    s, j = get(f"/knowledge/{kid2}", namespace=ns)
    check("get detail 200", s == 200 and j.get("id") == kid2, str(j))
    check("detail carries category", j.get("category") == "dev/backend", str(j))
    check("detail carries tags", j.get("tags") == "fastapi,python", str(j))

    s, j = get("/knowledge/99999", namespace=ns)
    check("get missing -> 404", s == 404, str(j))

    print("\n[4] category summary")
    s, j = get("/knowledge/categories", namespace=ns)
    check("categories 200", s == 200, str(j))
    cats = {x["category"] for x in j.get("categories", [])}
    check("has 3 categories", cats == {"ops/deploy", "dev/backend", "ops/proxy"}, str(cats))
    op = next((x for x in j.get("categories", []) if x["category"] == "ops/deploy"), None)
    check("ops/deploy has 1 entry", op and op["entries"] == 1, str(op))

    print("\n[5] delete single knowledge")
    s, j = delete(f"/knowledge/{kid3}", namespace=ns)
    check("delete knowledge 200", s == 200 and j.get("deleted") is True, str(j))
    s, j = get("/knowledge", namespace=ns)
    check("2 left after delete", len(j.get("results", [])) == 2, f"n={len(j.get('results', []))}")
    s, j = get(f"/knowledge/{kid3}", namespace=ns)
    check("deleted -> 404", s == 404, str(j))

    print("\n[6] knowledge-bases listing + delete")
    s, j = get("/knowledge-bases")
    kb = [x for x in j.get("knowledge_bases", []) if x["namespace"] == ns]
    check("kb listed", j.get("count", 0) >= 1 and kb and kb[0]["entries"] == 2, str(kb))

    s, j = delete(f"/knowledge-bases/{ns}")
    check("soft delete kb", s == 200 and j.get("hard") is False, str(j))

    srv.should_exit = True
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n===== RESULT: {PASS} passed, {FAIL} failed =====")
    sys.exit(1 if FAIL else 0)

if __name__ == "__main__":
    main()
