"""记忆 CRUD 路由。"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from engine.embeddings import EmbeddingClient
from engine.reflect import ReflectEngine, reflect_in_thread
from engine.retriever import HybridRetriever
from engine.store import HybridMemoryStore, _SHARD_CHARS

# 与 store 层分片阈值一致：超过此长度即不整篇 embed（改由分片逐片向量化）
_SHARD_CHARS_INLINE = _SHARD_CHARS

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memories"])


def _get_store_for_namespace(req: Request, namespace: str) -> HybridMemoryStore:
    settings = req.app.state.settings
    db_path = f"{settings.db_root}/{namespace}.db"
    store = HybridMemoryStore(db_path=db_path, embedding_dim=settings.embedding_dim)
    store.initialize()
    return store


def _reflect_in_thread(engine: ReflectEngine) -> dict:
    """兼容包装：实现已上提到 engine.reflect.reflect_in_thread。"""
    return reflect_in_thread(engine)


def _load_reflect_config(store: HybridMemoryStore, fallback: Settings) -> dict:
    """从 DB 加载用户设置的 reflect 配置，fallback 到环境变量默认值。"""
    try:
        store._conn.execute(
            "CREATE TABLE IF NOT EXISTS _config (key TEXT PRIMARY KEY, value TEXT)"
        )
        row = store._conn.execute(
            "SELECT value FROM _config WHERE key = 'reflect_config'"
        ).fetchone()
        if row:
            return json.loads(row[0])
    except Exception as e:
        logger.debug("read reflect config failed: %s", e)
    return {}


def _is_night_time(night_start: str, night_end: str) -> bool:
    """检查当前时间是否在夜间时段内。"""
    if not night_start or not night_end:
        return False  # 未配置时段 = 不限制
    import datetime
    now = datetime.datetime.now()
    start_parts = night_start.split(":")
    end_parts = night_end.split(":")
    try:
        start_min = int(start_parts[0]) * 60 + int(start_parts[1])
        end_min = int(end_parts[0]) * 60 + int(end_parts[1])
        now_min = now.hour * 60 + now.minute
        if start_min <= end_min:
            return start_min <= now_min <= end_min
        else:  # 跨天，如 22:00 - 08:00
            return now_min >= start_min or now_min <= end_min
    except (ValueError, IndexError):
        return False


class WriteRequest(BaseModel):
    content: str
    namespace: str = "default"
    memory_type: str = "observation"
    mem_action: str = ""
    mem_context: dict = Field(default_factory=dict)
    mem_outcome: dict = Field(default_factory=dict)
    mem_metadata: dict = Field(default_factory=dict)
    # 知识库导入：允许指定原始时间戳（如文档日期），缺省用当前时间
    created_at: str | None = None
    # 永久保留（pinned）：跳过自动去重合并，防被反思引擎误删
    pinned: bool = False
    # 文档溯源字段：带上即获得「不随时间衰减 + 不被建议遗忘」待遇，
    # 但 memory_type 保持不变（仍参与 reflect 提炼）—— 这是与整篇
    # POST /documents 分块入库的关键区别。用于课程/经典精读笔记等长文。
    doc_id: str | None = None
    doc_uri: str | None = None
    doc_title: str | None = None
    chunk_index: int | None = None
    doc_category: str | None = None
    doc_tags: list[str] = Field(default_factory=list)


class UpdateRequest(BaseModel):
    content: str


class Settings:
    """占位 — 实际运行时用 req.app.state.settings"""


@router.post("/memories")
async def write_memory(req: Request, body: WriteRequest):
    settings = req.app.state.settings

    # 复用进程级共享 client（见 main.py lifespan）：不能每请求新建，
    # 否则每次都重建 httpx 连接池、丢掉 keep-alive，每次写入白付一次建连。
    embedding_client = getattr(req.app.state, "embedding_client", None)
    if embedding_client is None and settings.embedding_base_url and settings.embedding_api_key:
        # 兼容未经过 lifespan 的场景（如单测直接构造 app）
        embedding_client = EmbeddingClient(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            embedding_model=settings.embedding_model,
            rerank_model=settings.rerank_model,
            embedding_dim=settings.embedding_dim,
        )

    embedding = None
    # 超长内容不要整篇 embed（必然超上游 8192 token 上限、白耗一次请求）：
    # 交给 add_memory_sharded 逐片向量化。
    #
    # ⚠ embed() 是同步 httpx 调用，必须扔线程池：直接 await 会阻塞 uvicorn
    # 唯一的事件循环。历史上一次写入因上游变慢卡住事件循环，连带所有请求
    # 排队 208 秒（同期 embed 本身只花 0.5s）。
    if embedding_client and len((body.content or "").strip()) <= _SHARD_CHARS_INLINE:
        embedding = await asyncio.to_thread(embedding_client.embed, body.content)

    store = _get_store_for_namespace(req, body.namespace)
    try:
        # 超长内容自动分片（v7）：父行保留全文，分片行带 shard_of 可检索、
        # 检索侧聚合回父行，避免单篇超长导致 rerank 整体 400。
        #
        # ⚠ 整个写入放线程池：分片路径会在 embed_fn 里做同步 HTTP 调用，
        # 直接在当前协程跑会把 uvicorn 事件循环卡住（连带其他请求排队）。
        # SQLite 也是同步 IO，同线程里一并更合适。
        def _do_write() -> tuple[int | None, int]:
            return store.add_memory_sharded(
                content=body.content,
                embed_fn=embedding_client.embed if embedding_client else None,
                embedding=embedding,
                memory_type=body.memory_type,
                mem_action=body.mem_action or None,
                mem_context=json.dumps(body.mem_context) if body.mem_context else None,
                mem_outcome=json.dumps(body.mem_outcome) if body.mem_outcome else None,
                mem_metadata=json.dumps(body.mem_metadata) if body.mem_metadata else None,
                created_at=body.created_at,
                # 溯源字段：doc_tags 为列表，落库前转逗号分隔（与 documents.py 一致）
                doc_id=body.doc_id or None,
                doc_uri=body.doc_uri or None,
                doc_title=body.doc_title or None,
                chunk_index=body.chunk_index,
                doc_category=body.doc_category or None,
                doc_tags=",".join(body.doc_tags) if body.doc_tags else None,
            )

        memory_id, shard_count = await asyncio.to_thread(_do_write)
        if memory_id is None:
            raise HTTPException(500, "Failed to store memory")
        if shard_count > 1:
            logger.info(
                "memory %d sharded into %d pieces (content %d chars)",
                memory_id,
                shard_count,
                len(body.content or ""),
            )

        # 永久保留（pinned）：立即打标，防反思引擎去重误删
        if body.pinned:
            try:
                store.set_pinned(memory_id, True)
            except Exception as e:
                logger.warning("set_pinned after write failed: %s", e)

        # 记录写入日志
        store.add_log(action="写入记忆", status="success", detail=f"类型: {body.memory_type}", namespace=body.namespace)

        # ─── 写入后后台自动触发 Reflect ───
        auto_reflected = False
        user_config = _load_reflect_config(store, settings)

        auto_enabled = user_config.get("auto_reflect", True)
        user_interval = int(user_config.get("interval_seconds", settings.reflect_interval))
        user_min_obs = int(user_config.get("min_observations", settings.reflect_min_observations))
        user_min_exp = int(user_config.get("min_experiences", settings.reflect_min_experiences))
        user_min_ins = int(user_config.get("min_insights", settings.reflect_min_insights))
        night_start = user_config.get("night_start", "")
        night_end = user_config.get("night_end", "")

        if auto_enabled:
            if night_start and night_end and not _is_night_time(night_start, night_end):
                logger.info("Skip auto-reflect: outside night window %s-%s", night_start, night_end)
            else:
                try:
                    llm_complete = None
                    if settings.embedding_base_url and settings.embedding_api_key:
                        from routers.reflect import _make_llm_complete
                        llm_complete = _make_llm_complete(
                            settings.embedding_base_url,
                            settings.embedding_api_key,
                            model=settings.reflect_model or "deepseek-v4-flash",
                        )

                    retriever = HybridRetriever(
                        store=store,
                        embedding_client=embedding_client,
                        keyword_weight=0.4,
                        vector_weight=0.6,
                    )

                    reflect_engine = ReflectEngine(
                        store=store,
                        retriever=retriever,
                        embedding_client=embedding_client,
                        min_experiences=user_min_exp,
                        min_observations=user_min_obs,
                        min_insights=user_min_ins,
                        reflection_interval=user_interval,
                        llm_complete=llm_complete,
                    )

                    if reflect_engine.should_reflect():
                        # 后台异步触发反思，不阻塞写入响应。
                        # 关键：不能复用请求级 store —— 本函数返回后 finally 会
                        # store.close()（_conn 置 None），而后台任务在响应之后才真正
                        # 执行，访问已关闭连接就报 "'NoneType' object has no attribute
                        # 'execute'"（生产日志累计 2383 次）。这里为后台任务自建独立
                        # store，与 main.py 的 _reflect_scheduler 同模式。
                        ns_bg = body.namespace
                        db_root = settings.db_root
                        emb_dim = settings.embedding_dim

                        async def _run_reflect_in_bg():
                            try:
                                # ⚠ store 的创建/使用/关闭必须全在同一个线程内完成。
                                # HybridMemoryStore 的 sqlite 连接是 check_same_thread=False
                                # 但并不线程安全；若在事件循环线程 close 而工作线程还在用，
                                # 会因并发操作同一连接而段错误（已实际复现）。
                                # 因此把整个生命周期交给 _reflect_in_thread 内部。
                                ns_local = ns_bg
                                db_root_local = db_root
                                dim_local = emb_dim
                                emb_local = embedding_client
                                llm_local = llm_complete
                                min_exp, min_obs, min_ins, interval = (
                                    user_min_exp, user_min_obs, user_min_ins, user_interval,
                                )

                                def _job() -> dict:
                                    bg_store = None
                                    try:
                                        bg_store = HybridMemoryStore(
                                            db_path=f"{db_root_local}/{ns_local}.db",
                                            embedding_dim=dim_local,
                                        )
                                        bg_store.initialize()
                                        bg_retriever = HybridRetriever(
                                            store=bg_store,
                                            embedding_client=emb_local,
                                            keyword_weight=0.4,
                                            vector_weight=0.6,
                                        )
                                        bg_engine = ReflectEngine(
                                            store=bg_store,
                                            retriever=bg_retriever,
                                            embedding_client=emb_local,
                                            min_experiences=min_exp,
                                            min_observations=min_obs,
                                            min_insights=min_ins,
                                            reflection_interval=interval,
                                            llm_complete=llm_local,
                                        )
                                        result = reflect_in_thread(bg_engine)
                                        stage = result.get("stage")
                                        if stage:
                                            logger.info(
                                                "Auto-reflect stage %s triggered after memory write (background)",
                                                stage,
                                            )
                                            counts = {k: v for k, v in result.get("counts", {}).items() if v}
                                            detail = f"阶段: {stage}"
                                            if counts:
                                                detail += ", " + ", ".join(f"{k}={v}" for k, v in counts.items())
                                            bg_store.add_log(
                                                action="自动反思",
                                                status="success",
                                                count=stage,
                                                detail=detail,
                                                namespace=ns_local,
                                            )
                                        return result
                                    finally:
                                        if bg_store is not None:
                                            bg_store.close()

                                # 反思循环含大量同步阻塞调用（dedup 的 merge_similar
                                # 逐条 embed、retriever 的同步 HTTP、同步 SQLite）。
                                # 直接在当前协程 await 会卡住 uvicorn 唯一的事件循环
                                # —— 历史事故：写入请求因此排队 208 秒。
                                await asyncio.to_thread(_job)
                            except Exception as e:
                                logger.warning("Background auto-reflect failed (non-fatal): %s", e)

                        asyncio.create_task(_run_reflect_in_bg())
                        auto_reflected = True
                except Exception as e:
                    logger.warning("Auto-reflect after write failed (non-fatal): %s", e)

        return {
            "memory_id": memory_id,
            "content": body.content[:100],
            "namespace": body.namespace,
            "embedded": embedding is not None,
            "auto_reflected": auto_reflected,
        }
    finally:
        store.close()


@router.get("/memories")
async def list_memories(
    req: Request,
    namespace: str | None = None,
    limit: int = 50,
    offset: int = 0,
    memory_type: str | None = None,
):
    namespace = namespace or "default"
    store = _get_store_for_namespace(req, namespace)
    try:
        results = store.list_memories(
            limit=min(limit, 200),
            offset=offset,
            memory_type=memory_type,
        )
        return {"results": results, "count": len(results)}
    finally:
        store.close()


@router.get("/memories/{memory_id}")
async def get_memory(
    req: Request,
    memory_id: int,
    namespace: str | None = None,
):
    if not namespace:
        raise HTTPException(400, "namespace query parameter is required")
    store = _get_store_for_namespace(req, namespace)
    try:
        memory = store.get_memory(memory_id)
        if not memory:
            raise HTTPException(404, "Memory not found")
        return memory
    finally:
        store.close()


@router.delete("/memories/{memory_id}")
async def delete_memory(
    req: Request,
    memory_id: int,
    namespace: str | None = None,
):
    if not namespace:
        raise HTTPException(400, "namespace query parameter is required")
    store = _get_store_for_namespace(req, namespace)
    try:
        ok = store.delete_memory(memory_id)
        return {"deleted": ok, "memory_id": memory_id}
    finally:
        store.close()
