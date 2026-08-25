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


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    namespace: str | None = None
    use_rerank: bool = True
    min_score: float | None = (
        None  # 最小相关度阈值；None = 用服务端默认（HMEM_MIN_SCORE），0 = 关闭过滤
    )
    extra_namespaces: list[str] | None = None


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

        # 分级共享：额外查询 shared 等共享库，按分数降序合并
        # （共享库是“用户偏好/心智模型/环境上下文”，业务语义仍主库为主，
        #  共享结果权重 ×0.8，避免喧宾夺主）
        extra = body.extra_namespaces or []
        for ns in extra:
            ns = ns.strip()
            if not ns or ns == namespace:
                continue
            extra_path = f"{settings.db_root}/{ns}.db"
            if not os.path.exists(extra_path):
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
                            r["score"] = r["score"] * 0.8
                        r["shared"] = True
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
