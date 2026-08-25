"""混合检索路由。"""

from __future__ import annotations

import logging
import os

from engine.embeddings import EmbeddingClient
from engine.retriever import HybridRetriever
from engine.store import HybridMemoryStore
from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


class ExtraNamespaceRef(BaseModel):
    """扩展命名空间引用：ns + 权重。

    weight 用于分级共享（如 shared 记忆库 0.8）或知识库（可给 1.0+）。
    """

    ns: str
    weight: float = 1.0


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    namespace: str | None = None
    use_rerank: bool = True
    min_score: float | None = (
        None  # 最小相关度阈值；None = 用服务端默认（HMEM_MIN_SCORE），0 = 关闭过滤
    )
    # 兼容两种形态：["shared"] 或 [{"ns": "kb-eng", "weight": 1.0}]
    extra_namespaces: list[str | ExtraNamespaceRef] | None = None


def _normalize_extra(ns_list: list[str | ExtraNamespaceRef] | None) -> list[ExtraNamespaceRef]:
    """把 extra_namespaces 归一化为 (ns, weight) 列表。

    纯字符串 → weight=1.0（老客户端/老行为）；对象 → 用显式权重。
    """
    out: list[ExtraNamespaceRef] = []
    for item in ns_list or []:
        if isinstance(item, str):
            out.append(ExtraNamespaceRef(ns=item))
        else:
            out.append(item)
    return out


def _inject_source(result: dict) -> None:
    """若结果属于知识库文档 chunk，附加 source 溯源对象。

    便于上层直接引用 doc_id/uri/标题/块号，而不必自己解析 mem_metadata。
    """
    doc_id = result.get("doc_id")
    if doc_id:
        result["source"] = {
            "doc_id": doc_id,
            "title": result.get("doc_title") or "",
            "uri": result.get("doc_uri") or "",
            "chunk_index": result.get("chunk_index") or 0,
        }


@router.post("/search")
async def search(req: Request, body: SearchRequest):
    namespace = body.namespace or "default"
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"

    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()

    embedding_client = None
    if settings.embedding_base_url and settings.embedding_api_key:
        embedding_client = EmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            embedding_dim=settings.embedding_dim,
        )

    retriever = HybridRetriever(
        store=store,
        embedding_client=embedding_client,
        keyword_weight=0.4,
        vector_weight=0.6,
        hrr_weight=settings.hrr_weight,
    )

    try:
        results = retriever.search(
            query=body.query,
            limit=min(body.limit, 50),
            use_rerank=body.use_rerank,
            min_score=body.min_score
            if body.min_score is not None
            else settings.min_score,
        )
        # 去掉 namespace 字段（分库后无意义），标注来源
        for r in results:
            r.pop("namespace", None)
            r["_ns"] = namespace
            _inject_source(r)

        # 分级共享 / 知识库多库检索：额外查询其它 namespace，按权重合并
        extra = _normalize_extra(body.extra_namespaces)
        for ref in extra:
            ns = ref.ns.strip()
            if not ns or ns == namespace:
                continue
            extra_path = f"{settings.db_root}/{ns}.db"
            if not os.path.exists(extra_path):
                logger.debug("extra namespace %s missing (db not found), skip", ns)
                continue
            try:
                estore = HybridMemoryStore(
                    db_path=extra_path, embedding_dim=settings.embedding_dim
                )
                estore.initialize()
                try:
                    eret = HybridRetriever(
                        store=estore,
                        embedding_client=embedding_client,
                        keyword_weight=0.4,
                        vector_weight=0.6,
                        hrr_weight=settings.hrr_weight,
                    )
                    extra_results = eret.search(
                        query=body.query,
                        limit=max(min(body.limit, 20), 5),
                        use_rerank=body.use_rerank,
                        min_score=body.min_score
                        if body.min_score is not None
                        else settings.min_score,
                    )
                    for r in extra_results:
                        r.pop("namespace", None)
                        r["_ns"] = ns
                        if r.get("score") is not None:
                            r["score"] = r["score"] * ref.weight
                        r["extra_weight"] = ref.weight
                        r["shared"] = True
                        _inject_source(r)
                    results.extend(extra_results)
                finally:
                    estore.close()
            except OSError:
                logger.debug("extra namespace %s unreadable, skip", ns)

        # 全量合并排序
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        results = results[: min(body.limit, 50)]
        return {"results": results, "count": len(results)}
    finally:
        store.close()