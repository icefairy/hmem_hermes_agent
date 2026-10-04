"""统计路由。"""

from __future__ import annotations

import glob
import logging
import os

from fastapi import APIRouter, HTTPException, Request

from engine.store import HybridMemoryStore
from routers.protected import PROTECTED_NAMESPACES

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stats"])


def _sanitize_ns(name: str) -> str:
    """命名空间白名单：仅字母数字、中划线、下划线、点。防路径穿越。"""
    if not name:
        raise HTTPException(400, "namespace is required")
    if not all(c.isalnum() or c in "-_." for c in name) or name.startswith("."):
        raise HTTPException(400, f"invalid namespace: {name!r}")
    return name


@router.get("/stats")
async def stats(req: Request, namespace: str | None = None):
    namespace = namespace or "default"
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"

    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    try:
        total = store.count_memories()
        embedding_enabled = bool(
            settings.embedding_base_url and settings.embedding_api_key
        )
        return {
            "total_memories": total,
            "embedding_enabled": embedding_enabled,
            "hrr_count": store.count_hrr(),
            "retrieval_mode": "ai" if embedding_enabled else "local",
            "by_type": store.count_by_type(),
            "document_count": store.count_documents(),
            "namespace": namespace,
        }
    finally:
        store.close()


@router.get("/namespaces")
async def list_namespaces(req: Request):
    """扫描 db_root 下所有 *.db 文件，返回可用 namespace 列表。"""
    settings = req.app.state.settings
    pattern = os.path.join(settings.db_root, "*.db")
    files = sorted(glob.glob(pattern))
    namespaces = []
    for fp in files:
        ns = os.path.splitext(os.path.basename(fp))[0]
        store = HybridMemoryStore(db_path=fp, embedding_dim=settings.embedding_dim)
        store.initialize()
        try:
            total = store.count_memories()
            namespaces.append(
                {
                    "namespace": ns,
                    "total_memories": total,
                    "protected": ns in PROTECTED_NAMESPACES,
                }
            )
        finally:
            store.close()
    return {"namespaces": namespaces}


@router.delete("/namespaces/{namespace}")
async def delete_namespace(req: Request, namespace: str):
    """删除一个命名空间(即删除其 db 文件)。"""
    ns = _sanitize_ns(namespace)
    settings = req.app.state.settings
    db_path = os.path.join(settings.db_root, f"{ns}.db")
    resolved = os.path.realpath(db_path)
    root = os.path.realpath(settings.db_root)
    if not resolved.startswith(root + os.sep) and resolved != root:
        raise HTTPException(400, "invalid namespace path")
    if not os.path.isfile(resolved):
        raise HTTPException(404, f"namespace not found: {ns}")
    # 最后一道防线：禁止删除受保护命名空间，避免误删全部记忆 / 手工知识库
    if ns in PROTECTED_NAMESPACES:
        raise HTTPException(
            400,
            f"cannot delete protected namespace: '{ns}' "
            "(reserved by system; protected namespaces: "
            + ", ".join(sorted(PROTECTED_NAMESPACES))
            + ")",
        )
    # 打开写入一条删除日志，然后关闭再删文件，避免文件占用
    store = HybridMemoryStore(db_path=resolved, embedding_dim=settings.embedding_dim)
    store.initialize()
    try:
        store.add_log(action="删除命名空间", status="success", count=0, namespace=ns)
    finally:
        store.close()
    try:
        os.remove(resolved)
    except OSError as e:
        logger.warning("delete namespace %s db failed: %s", ns, e)
        raise HTTPException(500, f"failed to remove db file: {e}") from e
    return {"deleted": True, "namespace": ns, "db_removed": True}


@router.post("/backfill/hrr")
async def backfill_hrr(req: Request, namespace: str | None = None):
    """为所有缺失 HRR 向量的存量记忆批量回填（本地 numpy，无 API 依赖）。"""
    ns = namespace or "default"
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{ns}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    try:
        n = store.rebuild_hrr_vectors()
        total = store.count_hrr()
        return {"backfilled": n, "total_hrr": total, "namespace": ns}
    finally:
        store.close()


@router.post("/backfill/tokenization")
async def backfill_tokenization(req: Request, namespace: str | None = None):
    """为存量记忆重建 content_jieba（文本归一化后分词）+ 刷新 FTS。

    引入文本归一化（NFKC/全半角/大小写，借鉴 Engram tokenizer compression）后，
    旧数据的 token 仍是未归一的；本端点一次性重算，让 FTS 对同形异写也能命中。
    幂等：可重复执行。
    """
    ns = namespace or "default"
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{ns}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    try:
        n = store.rebuild_tokenization()
        return {"rebuilt": n, "namespace": ns}
    finally:
        store.close()
