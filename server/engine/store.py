"""SQLite-backed memory store with FTS5 full-text search and sqlite-vec vector storage.

Schema v3:
  memories           — core fact table (id, content, content_jieba, memory_type,
                       mem_action, mem_context, mem_outcome, mem_metadata, parent_id,
                       hit_count, created_at, updated_at)
  memories_fts       — FTS5 virtual table over content_jieba (Chinese-aware via jieba)
  vec_memories       — sqlite-vec virtual table storing embedding vectors (dim=1024 float32)
  memory_edges       — graph edges for knowledge graph / causal chains

Memory types (v3):
  observation  — raw notes, unprocessed observations (low-level, short-lived)
  experience   — concrete experiences with action/context/outcome (mid-level)
  insight      — distilled patterns / reusable heuristics (high-level, durable)
  mental_model — refined mental models from multiple insights (highest, permanent)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from contextlib import suppress
from pathlib import Path
from typing import Any

import jieba
import sqlite_vec

from engine.holographic import (
    bytes_to_phases,
    encode_text,
    hrr_available,
    phases_to_bytes,
    similarity,
)

logger = logging.getLogger(__name__)

_VEC_TABLE = "vec_memories"
_FTS_TABLE = "memories_fts"
_MAIN_TABLE = "memories"
_EDGE_TABLE = "memory_edges"
_LOG_TABLE = "operation_logs"
_HRR_TABLE = "hrr_memories"
_META_TABLE = "hmem_meta"

VALID_MEMORY_TYPES = {
    # 五层结构：
    # 1. SELF 层（不可遗忘·跨会话）
    "self_identity",      # 身份/价值观/认知图接口
    # 2. 锚点结构层
    "anchor",             # 关键事件锚点
    "mental_model",       # 心智模型
    # 3. 知识层
    "knowledge",          # 长期事实知识
    "insight",            # 洞见
    # 4. 经验层
    "experience",         # 结构化经验
    # 5. 情境层（短期·可衰减）
    "observation",        # 原始观察/工作记忆
}

_SCHEMA_V2_SQL = f"""
CREATE TABLE IF NOT EXISTS {_MAIN_TABLE} (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content       TEXT NOT NULL,
    content_jieba TEXT NOT NULL DEFAULT '',
    memory_type   TEXT NOT NULL DEFAULT 'experience',
    mem_action    TEXT DEFAULT '',
    mem_context   TEXT DEFAULT '{{}}',
    mem_outcome   TEXT DEFAULT '{{}}',
    mem_metadata  TEXT DEFAULT '{{}}',
    parent_id     INTEGER DEFAULT NULL,
    hit_count     INTEGER DEFAULT 0,
    doc_id        TEXT DEFAULT '',
    doc_uri       TEXT DEFAULT '',
    doc_title     TEXT DEFAULT '',
    chunk_index   INTEGER DEFAULT 0,
    doc_category  TEXT DEFAULT '',
    doc_tags      TEXT DEFAULT '',
    -- v5: 重要性评分 + 主动遗忘（dsh-memory 阶段1）
    importance    REAL DEFAULT 0.4,
    archived      INTEGER DEFAULT 0,
    pinned        INTEGER DEFAULT 0,
    last_hit_at   TEXT DEFAULT '',
    -- v6: 不可遗忘标记（SELF 层专用）
    no_forget     INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT '2026-01-01 00:00:00',
    updated_at    TEXT NOT NULL DEFAULT '2026-01-01 00:00:00'
);

CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE}
    USING fts5(
        content_jieba,
        content UNINDEXED,
        content={_MAIN_TABLE},
        content_rowid=id,
        tokenize='unicode61'
    );

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON {_MAIN_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}(rowid, content_jieba, content)
        VALUES (new.id, new.content_jieba, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON {_MAIN_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, content_jieba, content)
        VALUES ('delete', old.id, old.content_jieba, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON {_MAIN_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, content_jieba, content)
        VALUES ('delete', old.id, old.content_jieba, old.content);
    INSERT INTO {_FTS_TABLE}(rowid, content_jieba, content)
        VALUES (new.id, new.content_jieba, new.content);
END;

CREATE INDEX IF NOT EXISTS idx_memories_type ON {_MAIN_TABLE}(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON {_MAIN_TABLE}(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_parent ON {_MAIN_TABLE}(parent_id);

-- Graph edges
CREATE TABLE IF NOT EXISTS {_EDGE_TABLE} (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES {_MAIN_TABLE}(id),
    target_id   INTEGER NOT NULL REFERENCES {_MAIN_TABLE}(id),
    relation    TEXT NOT NULL DEFAULT 'similar',
    weight      REAL DEFAULT 1.0,
    created_at  TEXT NOT NULL DEFAULT '2026-01-01 00:00:00'
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON {_EDGE_TABLE}(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON {_EDGE_TABLE}(target_id);

-- Operation logs
CREATE TABLE IF NOT EXISTS {_LOG_TABLE} (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'success',
    count       INTEGER DEFAULT 0,
    detail      TEXT DEFAULT '',
    namespace   TEXT NOT NULL DEFAULT 'default',
    created_at  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_logs_created ON {_LOG_TABLE}(created_at DESC);

-- Holographic vectors (local-only, numpy, no model API)
CREATE TABLE IF NOT EXISTS {_HRR_TABLE} (
    memory_id    INTEGER PRIMARY KEY REFERENCES {_MAIN_TABLE}(id) ON DELETE CASCADE,
    hrr_vector   BLOB NOT NULL,
    dim          INTEGER NOT NULL DEFAULT 1024,
    created_at   TEXT NOT NULL DEFAULT '2026-01-01 00:00:00'
);
CREATE INDEX IF NOT EXISTS idx_hrr_memory ON {_HRR_TABLE}(memory_id);

-- Engine metadata (per-namespace kv: reflect timestamps, counters, ...)
CREATE TABLE IF NOT EXISTS {_META_TABLE} (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_MIGRATE_V1_TO_V2 = """
-- Add v2 columns if they don't exist (idempotent)
ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'experience';
ALTER TABLE memories ADD COLUMN mem_action TEXT DEFAULT '';
ALTER TABLE memories ADD COLUMN mem_context TEXT DEFAULT '{}';
ALTER TABLE memories ADD COLUMN mem_outcome TEXT DEFAULT '{}';
ALTER TABLE memories ADD COLUMN mem_metadata TEXT DEFAULT '{}';
ALTER TABLE memories ADD COLUMN parent_id INTEGER DEFAULT NULL;
ALTER TABLE memories ADD COLUMN hit_count INTEGER DEFAULT 0;
"""

_MIGRATE_V2_TO_V3 = """
-- v3: expand memory_type to include 'observation' and 'insight'.
-- SQLite uses loose typing, so no ALTER COLUMN needed — just add the index.
-- Existing 'experience' values remain valid.
"""
# The app layer in add_memory() validates memory_type against VALID_MEMORY_TYPES.

# v4: 知识库文档列。SQLite 不支持 ADD COLUMN IF NOT EXISTS，逐条执行幂等迁移：
# 已存在的列会抛出 duplicate column 错误，捕获后跳过即可。
_DOC_COLUMNS = [
    ("doc_id", "TEXT DEFAULT ''"),
    ("doc_uri", "TEXT DEFAULT ''"),
    ("doc_title", "TEXT DEFAULT ''"),
    ("chunk_index", "INTEGER DEFAULT 0"),
    ("doc_category", "TEXT DEFAULT ''"),
    ("doc_tags", "TEXT DEFAULT ''"),
]

_MIGRATE_V3_TO_V4 = [f"ALTER TABLE memories ADD COLUMN {name} {ddl}" for name, ddl in _DOC_COLUMNS]

# v5: 重要性评分 + 主动遗忘（dsh-memory 阶段1）; v6: 不可遗忘标记（阶段3）。
# 与 v4 相同策略：逐条 ALTER，已存在列抛 duplicate column 错误，捕获后跳过（幂等）。
_MIGRATE_V4_TO_V6 = [
    "ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.4",
    "ALTER TABLE memories ADD COLUMN archived INTEGER DEFAULT 0",
    "ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0",
    "ALTER TABLE memories ADD COLUMN last_hit_at TEXT DEFAULT ''",
    "ALTER TABLE memories ADD COLUMN no_forget INTEGER DEFAULT 0",
]


def _tokenize(text: str) -> str:
    if not text:
        return ""
    words = jieba.lcut(text.strip())
    return " ".join(words)


def _heuristic_importance(
    memory_type: str, mem_action: str | None = None, mem_context: str | None = None
) -> float:
    """写入时启发式重要性评分（0~1），借鉴 dsh-memory 五层分层思想。"""
    _TYPE_IMPORTANCE = {
        "self_identity": 0.95,
        "mental_model": 0.85,
        "anchor": 0.85,
        "insight": 0.75,
        "knowledge": 0.6,
        "experience": 0.5,
        "observation": 0.4,
    }
    base = _TYPE_IMPORTANCE.get(memory_type, 0.5)
    if memory_type == "experience" and mem_action == "code_generation":
        base = max(base, 0.6)
    elif memory_type == "experience" and mem_action == "debug":
        base = max(base, 0.55)
    return base


def _now() -> str:
    """Returns current CST (UTC+8) formatted timestamp."""
    utc_now = datetime.now(timezone.utc)
    cst_now = utc_now + timedelta(hours=8)
    return cst_now.strftime("%Y-%m-%d %H:%M:%S")


class HybridMemoryStore:
    """Thread-safe SQLite store with FTS5 + vec + graph indexes."""

    def __init__(
        self,
        db_path: str,
        embedding_dim: int = 1024,
    ) -> None:
        self._db_path = str(Path(db_path).expanduser().resolve())
        self._embedding_dim = embedding_dim
        self._lock = threading.Lock()
        # 连接由 initialize() 建立（sqlite-vec 扩展需在初始化时加载）；
        # ok之前恒为 None，pyright 按 Connection 类型处理，赋值处屏蔽警告
        self._conn: sqlite3.Connection = None  # type: ignore[assignment]

    @property
    def embedding_dim(self) -> int:
        """公开 embedding 维度（HRR 编码、向量存储共用同一 dim）。"""
        return self._embedding_dim

    def initialize(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Load sqlite-vec
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # v1→v2 migration (idempotent)
        from contextlib import suppress

        with suppress(Exception):
            self._conn.executescript(_MIGRATE_V1_TO_V2)

        with suppress(Exception):
            self._conn.executescript(_MIGRATE_V2_TO_V3)

        # Create v2 schema (CREATE IF NOT EXISTS — idempotent)
        self._conn.executescript(_SCHEMA_V2_SQL)

        # v3→v4 迁移：逐条 ALTER（幂等，已存在列报错跳过）
        from contextlib import suppress as _suppress

        for _stmt in _MIGRATE_V3_TO_V4:
            with _suppress(Exception):
                self._conn.execute(_stmt)
        # v4→v6 迁移：v5/v6 新增列（幂等，与 v4 同一模式）
        for _stmt in _MIGRATE_V4_TO_V6:
            with _suppress(Exception):
                self._conn.execute(_stmt)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_doc ON memories(doc_id)"
        )
        self._conn.commit()

        # Create vec virtual table
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {_VEC_TABLE}"
            f" USING vec0("
            f"     memory_id INTEGER PRIMARY KEY,"
            f"     embedding float[{self._embedding_dim}]"
            f" )"
        )
        self._conn.commit()

    # -- Write operations ---------------------------------------------------

    def add_memory(
        self,
        content: str,
        embedding: list[float] | None = None,
        memory_type: str = "experience",
        mem_action: str | None = None,
        mem_context: str | None = None,
        mem_outcome: str | None = None,
        mem_metadata: str | None = None,
        parent_id: int | None = None,
        created_at: str | None = None,
        compute_hrr: bool = True,
        doc_id: str | None = None,
        doc_uri: str | None = None,
        doc_title: str | None = None,
        chunk_index: int | None = None,
        doc_category: str | None = None,
        doc_tags: str | None = None,
        importance: float | None = None,
    ) -> int | None:
        if not content or not content.strip():
            return None
        # Validate memory_type
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = "experience"
        content_jieba = _tokenize(content)
        # Heuristic importance if not explicitly provided
        if importance is None:
            importance = _heuristic_importance(memory_type, mem_action, mem_context)
        # Python-side timestamp to avoid SQLite strftime %% issues
        ts = created_at or _now()
        with self._lock:
            try:
                cur = self._conn.execute(
                    f"INSERT INTO {_MAIN_TABLE} "
                    f"(content, content_jieba, memory_type, "
                    f" mem_action, mem_context, mem_outcome, mem_metadata, parent_id, created_at, updated_at, "
                    f" doc_id, doc_uri, doc_title, chunk_index, doc_category, doc_tags, importance) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        content.strip(),
                        content_jieba,
                        memory_type,
                        mem_action or "",
                        mem_context or "{}",
                        mem_outcome or "{}",
                        mem_metadata or "{}",
                        parent_id,
                        ts,
                        ts,
                        doc_id or "",
                        doc_uri or "",
                        doc_title or "",
                        chunk_index or 0,
                        doc_category or "",
                        doc_tags or "",
                        importance,
                    ),
                )
                memory_id = cur.lastrowid
                if memory_id and embedding is not None:
                    try:
                        self._conn.execute(
                            f"INSERT INTO {_VEC_TABLE}(memory_id, embedding) VALUES (?, ?)",
                            (memory_id, json.dumps(embedding)),
                        )
                    except Exception as e:
                        logger.warning("vec insert failed for %d: %s", memory_id, e)
                if memory_id and compute_hrr:
                    self._save_hrr_vector(memory_id, content, ts)
                self._conn.commit()
                return memory_id
            except Exception as e:
                logger.error("add_memory failed: %s", e)
                self._conn.rollback()
                return None

    # -- Holographic (HRR) storage -------------------------------------------

    def _save_hrr_vector(
        self, memory_id: int, content: str, ts: str | None = None
    ) -> bool:
        """Compute the local HRR phase vector for a memory and store it.

        Local-only (numpy), no model API required. Failures are non-fatal —
        the memory itself still gets written.
        """
        if not hrr_available():
            return False
        try:
            vector = encode_text(content, self._embedding_dim)
            self._conn.execute(
                f"INSERT OR REPLACE INTO {_HRR_TABLE}(memory_id, hrr_vector, dim, created_at) "
                f"VALUES (?, ?, ?, ?)",
                (
                    memory_id,
                    phases_to_bytes(vector, self._embedding_dim),
                    self._embedding_dim,
                    ts or _now(),
                ),
            )
            return True
        except Exception as e:
            logger.warning("hrr encode failed for %d: %s", memory_id, e)
            return False

    def set_hrr_vector(self, memory_id: int, vector: Any) -> bool:
        """显式写入 HRR 向量（如重建/回填场景）。"""
        try:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {_HRR_TABLE}(memory_id, hrr_vector, dim, created_at) "
                f"VALUES (?, ?, ?, ?)",
                (
                    memory_id,
                    phases_to_bytes(vector, self._embedding_dim),
                    self._embedding_dim,
                    _now(),
                ),
            )
            self._conn.commit()
            return True
        except Exception as e:
            logger.warning("set_hrr_vector failed for %d: %s", memory_id, e)
            return False

    def get_hrr_vector(self, memory_id: int) -> Any | None:
        """取回单条记忆的 HRR 相位向量（未上线时返回 None）。"""
        try:
            row = self._conn.execute(
                f"SELECT hrr_vector, dim FROM {_HRR_TABLE} WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if not row:
                return None
            return bytes_to_phases(row[0], dim=row[1] or self._embedding_dim)
        except Exception as e:
            logger.debug("get_hrr_vector %d: %s", memory_id, e)
            return None

    def search_hrr(
        self,
        query_vec: Any,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """全库 HRR 相位相似度搜索（本地 numpy，无模型 API）。

        返回带 hrr_similarity（[0,1]）的候选列表，用于无网络/无 API key 场景的
        语义兜底检索。
        """
        if not hrr_available():
            return []
        rows = self._conn.execute(
            f"SELECT h.memory_id, h.hrr_vector, h.dim "
            f"FROM {_HRR_TABLE} h ORDER BY h.memory_id"
        ).fetchall()
        if not rows:
            return []

        results: list[tuple[int, float]] = []
        for r in rows:
            try:
                vec = bytes_to_phases(r[1], dim=r[2] or self._embedding_dim)
                sim = similarity(query_vec, vec)  # [-1,1]，0=无关
                sim01 = max(0.0, sim)  # 无关记忆=0，避免 0.5 基线噪声
                results.append((r[0], sim01))
            except Exception as e:
                logger.debug("hrr decode row failed: %s", e)

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:limit]
        if not top:
            return []
        ids = [t[0] for t in top]
        placeholders = ",".join("?" * len(ids))
        mem_rows = self._conn.execute(
            f"SELECT id, content, memory_type, created_at, updated_at, "
            f"       doc_id, doc_uri, doc_title, chunk_index "
            f"FROM {_MAIN_TABLE} WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        mem_map = {r[0]: r for r in mem_rows}
        out: list[dict[str, Any]] = []
        for mid, sim01 in top:
            r = mem_map.get(mid)
            if not r:
                continue
            out.append(
                {
                    "id": r[0],
                    "content": r[1],
                    "memory_type": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                    "doc_id": r[5],
                    "doc_uri": r[6],
                    "doc_title": r[7],
                    "chunk_index": r[8],
                    "hrr_similarity": round(sim01, 4),
                }
            )
        return out

    def rebuild_hrr_vectors(self) -> int:
        """为所有缺 HRR 向量的记忆批量计算并回填（后台/迁移用）。"""
        rows = self._conn.execute(
            f"SELECT m.id, m.content FROM {_MAIN_TABLE} m "
            f"LEFT JOIN {_HRR_TABLE} h ON h.memory_id = m.id "
            f"WHERE h.memory_id IS NULL"
        ).fetchall()
        n = 0
        for r in rows:
            if self._save_hrr_vector(r[0], r[1]):
                n += 1
        if n:
            self._conn.commit()
        return n

    def count_hrr(self) -> int:
        try:
            return int(
                self._conn.execute(f"SELECT COUNT(*) FROM {_HRR_TABLE}").fetchone()[0]
            )
        except Exception:
            return 0

    def add_edge(
        self,
        source_id: int,
        target_id: int,
        relation: str = "similar",
        weight: float = 1.0,
    ) -> bool:
        """在两条记忆之间创建关联边。"""
        with self._lock:
            try:
                self._conn.execute(
                    f"INSERT OR IGNORE INTO {_EDGE_TABLE} "
                    f"(source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)",
                    (source_id, target_id, relation, weight),
                )
                self._conn.commit()
                return True
            except Exception as e:
                logger.warning("add_edge failed: %s", e)
                return False

    def get_neighbors(
        self,
        memory_id: int,
        relation: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """取一条记忆在图谱中的一跳邻居（含边方向与关系类型）。

        Returns:
            [{"id", "content", "memory_type", "relation", "direction"}, ...]
            direction: "out" (source→target, 本记忆是上游) | "in" (本记忆是下游)
        """
        out: list[dict[str, Any]] = []
        with self._lock:
            try:
                rel_sql = "AND relation = ?" if relation else ""
                params: list[Any] = [memory_id] if not relation else [memory_id, relation]
                # 出边：本记忆为 source
                rows = self._conn.execute(
                    f"SELECT e.target_id, m.content, m.memory_type, e.relation "
                    f"FROM {_EDGE_TABLE} e "
                    f"JOIN {_MAIN_TABLE} m ON m.id = e.target_id "
                    f"WHERE e.source_id = ? {rel_sql} LIMIT ?",
                    (*params, limit),
                ).fetchall()
                for r in rows:
                    out.append(
                        {
                            "id": r[0],
                            "content": r[1],
                            "memory_type": r[2],
                            "relation": r[3],
                            "direction": "out",
                        }
                    )
                # 入边：本记忆为 target
                params_in: list[Any] = [memory_id] if not relation else [memory_id, relation]
                rows = self._conn.execute(
                    f"SELECT e.source_id, m.content, m.memory_type, e.relation "
                    f"FROM {_EDGE_TABLE} e "
                    f"JOIN {_MAIN_TABLE} m ON m.id = e.source_id "
                    f"WHERE e.target_id = ? {rel_sql} LIMIT ?",
                    (*params_in, limit),
                ).fetchall()
                for r in rows:
                    out.append(
                        {
                            "id": r[0],
                            "content": r[1],
                            "memory_type": r[2],
                            "relation": r[3],
                            "direction": "in",
                        }
                    )
                return out
            except Exception as e:
                logger.warning("get_neighbors failed: %s", e)
                return []

    def update_memory(
        self,
        memory_id: int,
        content: str,
        embedding: list[float] | None = None,
    ) -> bool:
        if not content or not content.strip():
            return False
        content_jieba = _tokenize(content)
        with self._lock:
            try:
                self._conn.execute(
                    f"UPDATE {_MAIN_TABLE} SET content=?, content_jieba=?, "
                    f"updated_at=? WHERE id=?",
                    (content.strip(), content_jieba, _now(), memory_id),
                )
                if self._conn.total_changes == 0:
                    return False
                if embedding is not None:
                    try:
                        # sqlite-vec 的 vec0 虚拟表不支持 INSERT OR REPLACE 的 REPLACE 语义
                        # （对已存在行会抛 UNIQUE constraint failed），改为先 DELETE 再 INSERT，
                        # 实现幂等写入（与 delete_memory 的 DELETE WHERE memory_id 一致）。
                        self._conn.execute(
                            f"DELETE FROM {_VEC_TABLE} WHERE memory_id = ?",
                            (memory_id,),
                        )
                        self._conn.execute(
                            f"INSERT INTO {_VEC_TABLE}(memory_id, embedding) VALUES (?, ?)",
                            (memory_id, json.dumps(embedding)),
                        )
                    except Exception as e:
                        logger.warning("vec update failed for %d: %s", memory_id, e)
                self._conn.commit()
                return True
            except Exception as e:
                logger.error("update_memory %d failed: %s", memory_id, e)
                self._conn.rollback()
                return False

    def increment_hit(self, memory_id: int) -> None:
        """增加记忆的命中计数。"""
        with self._lock:
            try:
                self._conn.execute(
                    f"UPDATE {_MAIN_TABLE} SET hit_count = hit_count + 1 WHERE id = ?",
                    (memory_id,),
                )
                self._conn.commit()
            except Exception as e:
                logger.debug("increment_hit %d failed: %s", memory_id, e)

    def _protected_ids(self, ids: list[int]) -> set[int]:
        """返回其中受保护（pinned / no_forget）的 id 集合。调用方必须已持有 _lock。

        统一保护判据的单点来源 —— 任何【硬删除 / 归档】路径都必须先过这里。
        2026-09-12 事故：保护只挂在 merge_similar 前置过滤上，属于补丁式防护，
        delete_by_doc_id 等路径可绕过。现下沉到 store 层，杜绝绕过。

        注：本方法**不自己取锁**（_lock 为非重入的 threading.Lock，重入会死锁）。
        """
        if not ids:
            return set()
        try:
            ph = ",".join("?" for _ in ids)
            rows = self._conn.execute(
                f"SELECT id FROM {_MAIN_TABLE} "
                f"WHERE id IN ({ph}) AND (pinned = 1 OR no_forget = 1)",
                ids,
            ).fetchall()
            return {r[0] for r in rows}
        except Exception as e:
            logger.warning("_protected_ids failed: %s", e)
            return set(ids)  # 查询失败时保守处理：全部视为受保护

    def delete_memory(self, memory_id: int) -> bool:
        """删除记忆。

        安全红线：pinned / no_forget 记忆拒绝删除。这是「永久保留」的最终兜底 ——
        2026-09-12 事故中去重引擎绕过一切保护直接把课程笔记删了，此处加锁防止
        未来任何新代码路径再次绕过。需要删除请先 set_pinned(False)。
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    f"SELECT pinned, no_forget FROM {_MAIN_TABLE} WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                if row and (row[0] or row[1]):
                    logger.warning(
                        "delete_memory %d refused: pinned=%s no_forget=%s（受保护）",
                        memory_id, row[0], row[1],
                    )
                    return False
                self._conn.execute(
                    f"DELETE FROM {_VEC_TABLE} WHERE memory_id = ?", (memory_id,)
                )
                self._conn.execute(
                    f"DELETE FROM {_EDGE_TABLE} WHERE source_id=? OR target_id=?",
                    (memory_id, memory_id),
                )
                self._conn.execute(
                    f"DELETE FROM {_MAIN_TABLE} WHERE id = ?", (memory_id,)
                )
                self._conn.commit()
                return self._conn.total_changes > 0
            except Exception as e:
                logger.error("delete_memory %d failed: %s", memory_id, e)
                self._conn.rollback()
                return False

    # -- Document-level operations (知识库) ------------------------------

    def delete_by_doc_id(self, doc_id: str) -> int:
        """级联删除某个文档的所有 chunk（含向量/边/FTS），返回删除条数。

        安全：pinned / no_forget 的 chunk 会被跳过（不删），受保护的 chunk 数量
        不影响返回值以外的行为。整篇文档被保护时本调用等于无操作。
        """
        doc_id = (doc_id or "").strip()
        if not doc_id:
            return 0
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT id FROM {_MAIN_TABLE} WHERE doc_id = ?", (doc_id,)
                ).fetchall()
                ids = [r[0] for r in rows]
                if not ids:
                    return 0
                # 保护下沉：受保护的 chunk 不参与删除
                protected = self._protected_ids(ids)
                if protected:
                    logger.warning(
                        "delete_by_doc_id %s: 跳过 %d 个受保护 chunk（pinned/no_forget）: %s",
                        doc_id, len(protected), sorted(protected)[:10],
                    )
                    ids = [i for i in ids if i not in protected]
                if not ids:
                    logger.warning(
                        "delete_by_doc_id %s refused: 全部 %d 个 chunk 均受保护",
                        doc_id, len(protected),
                    )
                    return 0
                ph = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"DELETE FROM {_VEC_TABLE} WHERE memory_id IN ({ph})", ids
                )
                self._conn.execute(
                    f"DELETE FROM {_EDGE_TABLE} WHERE source_id IN ({ph}) OR target_id IN ({ph})",
                    (*ids, *ids),
                )
                for mid in ids:  # FTS 由触发器级联删除
                    self._conn.execute(f"DELETE FROM {_MAIN_TABLE} WHERE id = ?", (mid,))
                self._conn.commit()
                return len(ids)
            except Exception as e:
                logger.error("delete_by_doc_id %s failed: %s", doc_id, e)
                self._conn.rollback()
                return 0

    def list_documents(self) -> list[dict[str, Any]]:
        """汇总所有文档：doc_id / 标题 / uri / 分类 / chunk 数 / 创建时间。"""
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT doc_id, doc_title, doc_uri, doc_category, doc_tags, "
                    f"       COUNT(*), MAX(created_at) "
                    f"FROM {_MAIN_TABLE} "
                    f"WHERE doc_id != '' GROUP BY doc_id ORDER BY MAX(created_at) DESC"
                ).fetchall()
                return [
                    {
                        "doc_id": r[0],
                        "doc_title": r[1],
                        "doc_uri": r[2],
                        "category": r[3] or "",
                        "tags": r[4] or "",
                        "chunk_count": r[5],
                        "created_at": r[6],
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.error("list_documents failed: %s", e)
                return []

    def count_documents(self) -> int:
        try:
            return int(
                self._conn.execute(
                    f"SELECT COUNT(DISTINCT doc_id) FROM {_MAIN_TABLE} WHERE doc_id != ''"
                ).fetchone()[0]
            )
        except Exception:
            return 0

    def category_stats(self) -> list[dict[str, Any]]:
        """知识库分类汇总：各分类的条目数 / 文档数 / 标签集合。"""
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT doc_category, COUNT(*), COUNT(DISTINCT doc_id), "
                    f"       GROUP_CONCAT(DISTINCT doc_tags) "
                    f"FROM {_MAIN_TABLE} WHERE doc_category != '' GROUP BY doc_category "
                    f"ORDER BY COUNT(*) DESC"
                ).fetchall()
                out = []
                for r in rows:
                    tags = set()
                    for tg in (r[3] or "").split(","):
                        if tg:
                            tags.add(tg)
                    out.append(
                        {
                            "category": r[0],
                            "entries": r[1],
                            "documents": r[2],
                            "tags": sorted(tags),
                        }
                    )
                return out
            except Exception as e:
                logger.error("category_stats failed: %s", e)
                return []

    # -- Read operations ----------------------------------------------------

    def list_memories(
        self,
        limit: int = 50,
        offset: int = 0,
        memory_type: str | None = None,
        category: str | None = None,
        doc_id: str | None = None,
        tags: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            try:
                where_parts = []
                params: list[Any] = []
                if memory_type:
                    where_parts.append("memory_type = ?")
                    params.append(memory_type)
                if category:
                    where_parts.append("doc_category = ?")
                    params.append(category)
                if doc_id:
                    where_parts.append("doc_id = ?")
                    params.append(doc_id)
                if tags:
                    # 逗号分隔标签任一命中即可（LIKE 匹配任一 tag）
                    tag_clauses = []
                    for t in tags.split(","):
                        t = t.strip()
                        if t:
                            tag_clauses.append("doc_tags LIKE ?")
                            params.append(f"%{t}%")
                    if tag_clauses:
                        where_parts.append("(" + " OR ".join(tag_clauses) + ")")
                where = "WHERE " + " AND ".join(where_parts) if where_parts else ""
                rows = self._conn.execute(
                    f"SELECT id, content, content_jieba, memory_type, "
                    f"  mem_action, mem_context, mem_outcome, mem_metadata, parent_id, "
                    f"  hit_count, created_at, updated_at, "
                    f"  doc_id, doc_uri, doc_title, chunk_index, doc_category, doc_tags, "
                    f"  importance, archived, pinned, last_hit_at, no_forget "
                    f"FROM {_MAIN_TABLE} {where} "
                    f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (*params, limit, offset),
                ).fetchall()
                return [self._row_to_dict(r) for r in rows]
            except Exception as e:
                logger.error("list_memories failed: %s", e)
                return []

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    f"SELECT id, content, content_jieba, memory_type, "
                    f"  mem_action, mem_context, mem_outcome, mem_metadata, parent_id, "
                    f"  hit_count, created_at, updated_at, "
                    f"  doc_id, doc_uri, doc_title, chunk_index, doc_category, doc_tags, "
                    f"  importance, archived, pinned, last_hit_at, no_forget "
                    f"FROM {_MAIN_TABLE} WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                return self._row_to_dict(row) if row else None
            except Exception:
                return None

    def get_child_memories(self, parent_id: int) -> list[dict[str, Any]]:
        """获取关联到某个心智模型的所有子经验。"""
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT m.id, m.content, m.memory_type, "
                    f"  m.created_at, m.updated_at "
                    f"FROM {_EDGE_TABLE} e "
                    f"JOIN {_MAIN_TABLE} m ON e.source_id = m.id "
                    f"WHERE e.target_id = ? AND e.relation = 'supporting_evidence' "
                    f"ORDER BY m.created_at DESC LIMIT 100",
                    (parent_id,),
                ).fetchall()
                return [
                    {
                        "id": r[0],
                        "content": r[1],
                        "memory_type": r[2],
                        "created_at": r[3],
                        "updated_at": r[4],
                    }
                    for r in rows
                ]
            except Exception:
                return []

    def search_fts(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        tokenized = _tokenize(query)
        if not tokenized.strip():
            tokenized = query.strip()

        fts_parts = []
        for t in tokenized.split():
            t = t.strip()
            if t:
                fts_parts.append(f'"{t}"*')
        fts_query = " OR ".join(fts_parts) if fts_parts else query

        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT id, content, memory_type, created_at, updated_at, "
                    f"       doc_id, doc_uri, doc_title, chunk_index, rank "
                    f"FROM {_FTS_TABLE} f "
                    f"JOIN {_MAIN_TABLE} m ON f.rowid = m.id "
                    f"WHERE {_FTS_TABLE} MATCH ? "
                    f"ORDER BY rank LIMIT ?",
                    (fts_query, limit),
                ).fetchall()
                if rows:
                    results = []
                    for r in rows:
                        d = {
                            "id": r[0],
                            "content": r[1],
                            "memory_type": r[2],
                            "created_at": r[3],
                            "updated_at": r[4],
                        }
                        d["fts_rank"] = r[9]
                        if r[5]:
                            d["doc_id"] = r[5]
                            d["doc_uri"] = r[6]
                            d["doc_title"] = r[7]
                            d["chunk_index"] = r[8]
                        results.append(d)
                    return results
            except Exception as e:
                logger.debug("FTS5 failed: %s", e)

            # LIKE fallback
            try:
                like = f"%{query.strip()}%"
                rows = self._conn.execute(
                    f"SELECT id, content, memory_type, "
                    f"  created_at, updated_at, doc_id, doc_uri, doc_title, chunk_index "
                    f"FROM {_MAIN_TABLE} "
                    f"WHERE content LIKE ? "
                    f"ORDER BY created_at DESC LIMIT ?",
                    (like, limit),
                ).fetchall()

                if not rows:
                    key_tokens = [t for t in tokenized.split() if len(t) > 1]
                    for token in key_tokens[:5]:
                        like = f"%{token}%"
                        r = self._conn.execute(
                            f"SELECT id, content, memory_type, "
                            f"  created_at, updated_at, doc_id, doc_uri, doc_title, chunk_index "
                            f"FROM {_MAIN_TABLE} "
                            f"WHERE content LIKE ? "
                            f"ORDER BY created_at DESC LIMIT ?",
                            (like, limit),
                        ).fetchall()
                        rows.extend(r)
                        if len(rows) >= limit:
                            break

                    seen = set()
                    deduped = []
                    for r in rows:
                        if r[0] not in seen:
                            seen.add(r[0])
                            deduped.append(r)
                    rows = deduped[:limit]

                return [
                    {
                        "id": r[0],
                        "content": r[1],
                        "memory_type": r[2],
                        "created_at": r[3],
                        "updated_at": r[4],
                        "fts_rank": -1.0,
                        **(
                            {
                                "doc_id": r[5],
                                "doc_uri": r[6],
                                "doc_title": r[7],
                                "chunk_index": r[8],
                            }
                            if r[5]
                            else {}
                        ),
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.debug("LIKE fallback failed: %s", e)
                return []

    def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not embedding:
            return []
        embedding_json = json.dumps(embedding)
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT m.id, m.content, m.memory_type, "
                    f"  m.created_at, m.updated_at, m.doc_id, m.doc_uri, m.doc_title, m.chunk_index, v.distance "
                    f"FROM {_VEC_TABLE} v "
                    f"JOIN {_MAIN_TABLE} m ON v.memory_id = m.id "
                    f"WHERE v.embedding MATCH ? "
                    f"ORDER BY v.distance LIMIT ?",
                    (embedding_json, limit),
                ).fetchall()
                results = []
                for r in rows:
                    d = {
                        "id": r[0],
                        "content": r[1],
                        "memory_type": r[2],
                        "created_at": r[3],
                        "updated_at": r[4],
                    }
                    d["vec_distance"] = float(r[5])
                    d["vec_similarity"] = 1.0 / (1.0 + float(r[5]))
                    results.append(d)
                return results
            except Exception as e:
                logger.debug("Vector search failed: %s", e)
                return []

    def count_memories(self, memory_type: str | None = None) -> int:
        with self._lock:
            try:
                if memory_type:
                    row = self._conn.execute(
                        f"SELECT COUNT(*) FROM {_MAIN_TABLE} WHERE memory_type = ?",
                        (memory_type,),
                    ).fetchone()
                else:
                    row = self._conn.execute(
                        f"SELECT COUNT(*) FROM {_MAIN_TABLE}"
                    ).fetchone()
                return row[0] if row else 0
            except Exception:
                return 0

    def count_by_type(self) -> dict[str, int]:
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT memory_type, COUNT(*) FROM {_MAIN_TABLE} "
                    f"GROUP BY memory_type"
                ).fetchall()
                return {r[0]: r[1] for r in rows}
            except Exception:
                return {}

    # -- Engine metadata (kv) ------------------------------------------------

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        """读一条引擎元数据（per-namespace，存于 hmem_meta 表）。"""
        with self._lock:
            try:
                row = self._conn.execute(
                    f"SELECT value FROM {_META_TABLE} WHERE key = ?", (key,)
                ).fetchone()
                return row[0] if row else default
            except Exception:
                return default

    def set_meta(self, key: str, value: str) -> None:
        """写一条引擎元数据（UPSERT）。"""
        with self._lock:
            try:
                self._conn.execute(
                    f"INSERT INTO {_META_TABLE}(key, value) VALUES (?, ?) "
                    f"ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
                self._conn.commit()
            except Exception as e:
                logger.warning("set_meta(%s) failed: %s", key, e)

    # -- Vector bulk read (dedup/reflect 复用已存向量，避免重复调 embed API) --

    def list_vectors(self, memory_type: str | None = None) -> dict[int, list[float]]:
        """全量读取已持久化的 embedding 向量。

        Args:
            memory_type: 限定记忆类型（None = 全部）。

        Returns:
            {memory_id: vector}；仅含 vec_memories 里已有向量的条目。
        """
        sql = f"SELECT v.memory_id, vec_to_json(v.embedding) FROM {_VEC_TABLE} v"
        params: list[Any] = []
        if memory_type is not None:
            sql += f" JOIN {_MAIN_TABLE} m ON v.memory_id = m.id WHERE m.memory_type = ?"
            params.append(memory_type)
        out: dict[int, list[float]] = {}
        with self._lock:
            try:
                for mid, emb in self._conn.execute(sql, params).fetchall():
                    try:
                        # vec_to_json 返回 JSON 字符串（sqlite-vec 内部以 float32 blob 存储）
                        vec = json.loads(emb) if isinstance(emb, (str, bytes)) else list(emb)
                        if isinstance(vec, list) and vec:
                            out[int(mid)] = [float(x) for x in vec]
                    except Exception as _e:
                        logger.debug("vec parse failed: %s", _e)
                        continue
            except Exception as e:
                logger.debug("list_vectors failed: %s", e)
        return out

    def upsert_vector(self, memory_id: int, embedding: list[float]) -> bool:
        """补写/覆盖一条 embedding 向量（去重回填用）。

        注意：sqlite-vec 虚拟表不支持 ON CONFLICT UPSERT，用 DELETE+INSERT。
        """
        with self._lock:
            try:
                self._conn.execute(
                    f"DELETE FROM {_VEC_TABLE} WHERE memory_id = ?", (memory_id,)
                )
                self._conn.execute(
                    f"INSERT INTO {_VEC_TABLE}(memory_id, embedding) VALUES (?, ?)",
                    (memory_id, json.dumps(embedding)),
                )
                self._conn.commit()
                return True
            except Exception as e:
                logger.warning("upsert_vector(%d) failed: %s", memory_id, e)
                return False

    # -- Edge bulk read -------------------------------------------------------

    def get_edge_source_ids(self, relation: str) -> set[int]:
        """取某关系类型全部出边的 source_id 集合（reflect 增量判断用）。"""
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT DISTINCT source_id FROM {_EDGE_TABLE} WHERE relation = ?",
                    (relation,),
                ).fetchall()
                return {int(r[0]) for r in rows}
            except Exception:
                return set()

    # -- Operation logs ----------------------------------------------------

    def add_log(
        self,
        action: str,
        status: str = "success",
        count: int = 0,
        detail: str = "",
        namespace: str = "default",
    ) -> int | None:
        ts = _now()
        with self._lock:
            try:
                cur = self._conn.execute(
                    f"INSERT INTO {_LOG_TABLE} (action, status, count, detail, namespace, created_at) "
                    f"VALUES (?, ?, ?, ?, ?, ?)",
                    (action, status, count, detail, namespace, ts),
                )
                self._conn.commit()
                return cur.lastrowid
            except Exception as e:
                logger.error("add_log failed: %s", e)
                return None

    def list_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        namespace: str | None = None,
    ) -> list[dict]:
        with self._lock:
            try:
                where = ""
                params: list = []
                if namespace:
                    where = "WHERE namespace = ?"
                    params.append(namespace)
                rows = self._conn.execute(
                    f"SELECT id, action, status, count, detail, namespace, created_at "
                    f"FROM {_LOG_TABLE} {where} "
                    f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (*params, limit, offset),
                ).fetchall()
                return [
                    {
                        "id": r[0],
                        "action": r[1],
                        "status": r[2],
                        "count": r[3],
                        "detail": r[4],
                        "namespace": r[5],
                        "created_at": r[6],
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.error("list_logs failed: %s", e)
                return []

    def get_graph(
        self,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """返回力导向图数据。limit=None 时返回全部节点。"""
        nodes: list[dict] = []
        edges: list[dict] = []
        with self._lock:
            try:
                if limit is not None:
                    rows = self._conn.execute(
                        f"SELECT id, content, memory_type, "
                        f"  hit_count FROM {_MAIN_TABLE} "
                        f"ORDER BY created_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT id, content, memory_type, "
                        f"  hit_count FROM {_MAIN_TABLE} "
                        f"ORDER BY created_at DESC",
                    ).fetchall()

                id_set = {r[0] for r in rows}
                nodes = [
                    {
                        "id": r[0],
                        "label": r[1][:60] + ("\u2026" if len(r[1]) > 60 else ""),
                        "type": r[2],
                        "hit_count": r[3],
                    }
                    for r in rows
                ]

                if id_set:
                    placeholders = ",".join("?" for _ in id_set)
                    edge_rows = self._conn.execute(
                        f"SELECT source_id, target_id, relation, weight "
                        f"FROM {_EDGE_TABLE} "
                        f"WHERE source_id IN ({placeholders}) "
                        f"AND target_id IN ({placeholders})",
                        (*id_set, *id_set),
                    ).fetchall()
                    edges = [
                        {
                            "source": r[0],
                            "target": r[1],
                            "relation": r[2],
                            "weight": r[3],
                        }
                        for r in edge_rows
                    ]
            except Exception as e:
                logger.debug("get_graph failed: %s", e)

        return {"nodes": nodes, "edges": edges}

    # -- Utils ---------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: tuple) -> dict[str, Any]:
        d = {
            "id": row[0],
            "content": row[1],
            "content_jieba": row[2],
            "memory_type": row[3],
            "mem_action": row[4],
            "mem_context": row[5],
            "mem_outcome": row[6],
            "mem_metadata": row[7],
            "parent_id": row[8],
            "hit_count": row[9],
            "created_at": row[10],
            "updated_at": row[11],
        }
        # v4: 知识库文档字段（无 doc_id 的记忆不带这些键，保持兼容）
        if len(row) > 12 and row[12]:
            d["doc_id"] = row[12]
            d["doc_uri"] = row[13]
            d["doc_title"] = row[14]
            d["chunk_index"] = row[15]
            if len(row) > 17:
                d["doc_category"] = row[16] or ""
                d["doc_tags"] = row[17] or ""
        # v5: importance, archived, pinned, last_hit_at
        if len(row) > 18:
            d["importance"] = row[18]
            d["archived"] = row[19]
            d["pinned"] = row[20]
            d["last_hit_at"] = row[21] or ""
        # v6: no_forget
        if len(row) > 22:
            d["no_forget"] = row[22]
        return d

    def close(self) -> None:
        conn = self._conn  # type: ignore[reportAttributeAccessIssue]  # close 前可能未 initialize，与 __init__ 注释一致
        if conn:
            with suppress(Exception):
                conn.close()
        self._conn = None  # type: ignore[assignment]  # 类注解非 Optional，close 后不再使用

    # -- Lifecycle methods (v5) ----------------------------------------------

    def mark_hit(self, memory_id: int) -> bool:
        """检索命中后联动：hit_count+1, last_hit_at更新, importance+0.02（上限0.99）。"""
        with self._lock:
            try:
                self._conn.execute(
                    f"UPDATE {_MAIN_TABLE} SET "
                    f"  hit_count = hit_count + 1, "
                    f"  last_hit_at = ?, "
                    f"  importance = MIN(importance + 0.02, 0.99), "
                    f"  updated_at = ? "
                    f"WHERE id = ?",
                    (_now(), _now(), memory_id),
                )
                self._conn.commit()
                return True
            except Exception as e:
                logger.error("mark_hit %d failed: %s", memory_id, e)
                self._conn.rollback()
                return False

    def get_importance_map(self, ids: list[int]) -> dict[int, float]:
        """批量读取一批记忆的重要性评分 {id: importance}。"""
        if not ids:
            return {}
        out: dict[int, float] = {}
        with self._lock:
            try:
                ph = ",".join("?" * len(ids))
                rows = self._conn.execute(
                    f"SELECT id, importance FROM {_MAIN_TABLE} WHERE id IN ({ph})",
                    list(ids),
                ).fetchall()
                for r in rows:
                    out[r[0]] = float(r[1] or 0.5)
            except Exception as e:
                logger.debug("get_importance_map failed: %s", e)
        return out

    def set_importance(self, memory_id: int, value: float) -> bool:
        """设置记忆重要性（0~1）。只升不降：新值低于当前值时忽略。"""
        try:
            value = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return False
        with self._lock:
            try:
                row = self._conn.execute(
                    f"SELECT importance FROM {_MAIN_TABLE} WHERE id = ?", (memory_id,)
                ).fetchone()
                if not row:
                    return False
                # 只升不降
                if value <= float(row[0]):
                    return True
                self._conn.execute(
                    f"UPDATE {_MAIN_TABLE} SET importance = ?, updated_at = ? "
                    f"WHERE id = ?",
                    (value, _now(), memory_id),
                )
                self._conn.commit()
                return True
            except Exception as e:
                logger.error("set_importance %d failed: %s", memory_id, e)
                self._conn.rollback()
                return False

    def set_pinned(self, memory_id: int, pinned: bool = True) -> bool:
        """标记 / 取消标记为永久保留（SELF 层 / 协议锚点，不可遗忘）。"""
        with self._lock:
            try:
                self._conn.execute(
                    f"UPDATE {_MAIN_TABLE} SET pinned = ?, updated_at = ? "
                    f"WHERE id = ?",
                    (1 if pinned else 0, _now(), memory_id),
                )
                self._conn.commit()
                return True
            except Exception as e:
                logger.error("set_pinned %d failed: %s", memory_id, e)
                self._conn.rollback()
                return False

    def set_no_forget(self, memory_id: int, no_forget: bool = True) -> bool:
        """标记/取消标记为不可遗忘（与 pinned 配合，专门用于 SELF 层）。"""
        with self._lock:
            try:
                self._conn.execute(
                    f"UPDATE {_MAIN_TABLE} SET no_forget = ?, updated_at = ? "
                    f"WHERE id = ?",
                    (1 if no_forget else 0, _now(), memory_id),
                )
                self._conn.commit()
                return True
            except Exception as e:
                logger.error("set_no_forget %d failed: %s", memory_id, e)
                self._conn.rollback()
                return False

    def archive_memory(self, memory_id: int) -> bool:
        """归档记忆（可逆主动遗忘）。pinned / no_forget 记忆不可归档。

        注：原实现只检查 pinned，漏了 no_forget —— SELF 层标记（no_forget）
        的记忆可能被归档。2026-09-12 统一为同一判据。
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    f"SELECT pinned, no_forget FROM {_MAIN_TABLE} WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                if not row or row[0] or row[1]:
                    return False
                cur = self._conn.execute(
                    f"UPDATE {_MAIN_TABLE} SET archived = 1, updated_at = ? "
                    f"WHERE id = ?",
                    (_now(), memory_id),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except Exception as e:
                logger.error("archive_memory %d failed: %s", memory_id, e)
                self._conn.rollback()
                return False

    def unarchive_memory(self, memory_id: int) -> bool:
        """从归档恢复记忆。"""
        with self._lock:
            try:
                cur = self._conn.execute(
                    f"UPDATE {_MAIN_TABLE} SET archived = 0, updated_at = ? "
                    f"WHERE id = ?",
                    (_now(), memory_id),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except Exception as e:
                logger.error("unarchive_memory %d failed: %s", memory_id, e)
                self._conn.rollback()
                return False

    def list_archived(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """列出已归档记忆。"""
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT id, content, memory_type, created_at, updated_at, "
                    f"  importance, archived, pinned FROM {_MAIN_TABLE} "
                    f"WHERE archived = 1 "
                    f"ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
                return [
                    {
                        "id": r[0],
                        "content": r[1][:200],
                        "memory_type": r[2],
                        "created_at": r[3],
                        "updated_at": r[4],
                        "importance": r[5],
                        "archived": r[6],
                        "pinned": r[7],
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.error("list_archived failed: %s", e)
                return []

    def forget_advisor(self, threshold_days: int = 60, min_importance: float = 0.4, max_candidates: int = 50) -> dict[str, Any]:
        """主动遗忘顾问：找出"长期未命中 + 低重要性 + 非 pinned + 非 no_forget + 非知识库"的记忆。"""
        import datetime as _dt
        cutoff = (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=8) - _dt.timedelta(days=threshold_days)).strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT id, content, memory_type, importance, hit_count, "
                    f"  created_at, updated_at, last_hit_at, archived, pinned "
                    f"FROM {_MAIN_TABLE} "
                    f"WHERE archived = 0 AND pinned = 0 AND no_forget = 0 "
                    f"  AND (doc_id = '' OR doc_id IS NULL) "
                    f"  AND importance < ? "
                    f"  AND (last_hit_at = '' OR last_hit_at < ?) "
                    f"ORDER BY importance ASC, updated_at ASC LIMIT ?",
                    (min_importance, cutoff, max_candidates),
                ).fetchall()
                candidates = [
                    {
                        "id": r[0],
                        "content": r[1][:120],
                        "memory_type": r[2],
                        "importance": r[3],
                        "hit_count": r[4],
                        "created_at": r[5],
                        "updated_at": r[6],
                    }
                    for r in rows
                ]
                total_idle = self._conn.execute(
                    f"SELECT COUNT(*) FROM {_MAIN_TABLE} "
                    f"WHERE archived = 0 AND pinned = 0 AND no_forget = 0 "
                    f"  AND (last_hit_at = '' OR last_hit_at < ?)",
                    (cutoff,),
                ).fetchone()[0]
                return {
                    "candidates": candidates,
                    "candidate_count": len(candidates),
                    "total_idle": total_idle,
                    "threshold_days": threshold_days,
                    "min_importance": min_importance,
                }
            except Exception as e:
                logger.error("forget_advisor failed: %s", e)
                return {"candidates": [], "candidate_count": 0, "total_idle": 0, "error": str(e)}

    def get_layer_stats(self) -> dict[str, Any]:
        """返回五层记忆分布统计。"""
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT memory_type, COUNT(*) FROM {_MAIN_TABLE} "
                    f"WHERE archived = 0 GROUP BY memory_type"
                ).fetchall()
                layer_dist = {r[0]: r[1] for r in rows}
                layers = {
                    "self": layer_dist.get("self_identity", 0),
                    "anchor": layer_dist.get("mental_model", 0) + layer_dist.get("anchor", 0),
                    "knowledge": layer_dist.get("insight", 0) + layer_dist.get("knowledge", 0),
                    "experience": layer_dist.get("experience", 0),
                    "context": layer_dist.get("observation", 0),
                }
                total = sum(layers.values()) or 1
                return {
                    **layers,
                    "total_active": total,
                    "layer_percentages": {k: round(v/total, 3) for k, v in layers.items()},
                }
            except Exception as e:
                logger.error("get_layer_stats failed: %s", e)
                return {}

    def flywheel_metrics(self) -> dict[str, Any]:
        """知识飞轮指标：增长率/复用率/蒸馏率。"""
        with self._lock:
            try:
                import datetime as _dt
                week_ago = (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=8) - _dt.timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
                total_row = self._conn.execute(f"SELECT COUNT(*) FROM {_MAIN_TABLE} WHERE archived = 0").fetchone()
                total = total_row[0] if total_row else 0
                new_row = self._conn.execute(
                    f"SELECT COUNT(*) FROM {_MAIN_TABLE} WHERE created_at >= ? AND archived = 0", (week_ago,)
                ).fetchone()
                new_count = new_row[0] if new_row else 0
                growth_rate = new_count / total if total > 0 else 0
                reused_row = self._conn.execute(
                    f"SELECT COUNT(*) FROM {_MAIN_TABLE} WHERE archived = 0 AND hit_count > 0"
                ).fetchone()
                reused = reused_row[0] if reused_row else 0
                reuse_rate = reused / total if total > 0 else 0
                insight_row = self._conn.execute(
                    f"SELECT COUNT(*) FROM {_MAIN_TABLE} WHERE memory_type IN ('insight', 'mental_model') AND archived = 0"
                ).fetchone()
                insight_count = insight_row[0] if insight_row else 0
                base_row = self._conn.execute(
                    f"SELECT COUNT(*) FROM {_MAIN_TABLE} WHERE memory_type IN ('experience', 'observation') AND archived = 0"
                ).fetchone()
                base_count = base_row[0] if base_row else 0
                distill_rate = insight_count / base_count if base_count > 0 else 0
                return {
                    "total_memories": total,
                    "new_last_7days": new_count,
                    "growth_rate": round(growth_rate, 4),
                    "reused_memories": reused,
                    "reuse_rate": round(reuse_rate, 4),
                    "distilled_count": insight_count,
                    "base_count": base_count,
                    "distill_rate": round(distill_rate, 4),
                }
            except Exception as e:
                logger.error("flywheel_metrics failed: %s", e)
                return {"error": str(e)}

    def cognition_report(self) -> dict[str, Any]:
        """自我认知报告（借鉴 dsh-memory P0 cognition / self_reliability）。"""
        with self._lock:
            try:
                row = self._conn.execute(f"SELECT COUNT(*) FROM {_MAIN_TABLE}").fetchone()
                total = int(row[0]) if row else 0
                if total == 0:
                    return {
                        "total_memories": 0, "reliability": "watch",
                        "hit_distribution": {}, "importance_distribution": {},
                        "archival_ratio": 0.0, "graph_snr": 0.0,
                        "layer_distribution": {},
                        "suggestions": ["暂无记忆，开始使用 hmem_write 建立记忆库"],
                    }
                hit_rows = self._conn.execute(
                    f"SELECT hit_count, COUNT(*) FROM {_MAIN_TABLE} WHERE archived = 0 GROUP BY hit_count ORDER BY hit_count"
                ).fetchall()
                hit_dist = {int(r[0]): r[1] for r in hit_rows}
                hit_zero = hit_dist.get(0, 0)
                hit_active = total - hit_zero
                imp_rows = self._conn.execute(
                    f"SELECT CASE WHEN importance < 0.4 THEN 'low' WHEN importance < 0.7 THEN 'med' ELSE 'high' END, COUNT(*) "
                    f"FROM {_MAIN_TABLE} WHERE archived = 0 GROUP BY 1"
                ).fetchall()
                imp_dist = {r[0]: r[1] for r in imp_rows}
                archived = self._conn.execute(
                    f"SELECT COUNT(*) FROM {_MAIN_TABLE} WHERE archived = 1"
                ).fetchone()[0]
                archival_ratio = archived / total if total > 0 else 0.0
                edge_rows = self._conn.execute(
                    f"SELECT relation, COUNT(*) FROM {_EDGE_TABLE} GROUP BY relation"
                ).fetchall()
                edge_map = {r[0]: r[1] for r in edge_rows}
                enriched = edge_map.get("enriched_to", 0)
                total_edges = sum(edge_map.values())
                graph_snr = enriched / total_edges if total_edges > 0 else 0.0
                active_ratio = hit_active / total if total > 0 else 0
                if archival_ratio > 0.5:
                    reliability = "degraded"
                elif active_ratio < 0.1 and total > 10:
                    reliability = "watch"
                else:
                    reliability = "reliable"
                suggestions: list[str] = []
                if hit_zero > total * 0.8 and total > 20:
                    suggestions.append("大量记忆未被命中，考虑运行 /hmem reflect 激活或归档低价值记忆")
                if archival_ratio > 0.3:
                    suggestions.append(f"归档率 {archival_ratio:.0%}，检查是否有过多记忆被误归档")
                if graph_snr < 0.5 and total_edges > 10:
                    suggestions.append("图谱 enriched_to 边比例偏低，鼓励创建更多经验→洞见的关联")
                if imp_dist.get("low", 0) > total * 0.6:
                    suggestions.append("大部分记忆重要性偏低，可考虑批量提升高价值经验")
                if not suggestions:
                    suggestions.append("记忆库健康，继续保持当前使用模式")
                # Inline layer stats to avoid deadlock (both use self._lock)
                try:
                    lrows = self._conn.execute(
                        f"SELECT memory_type, COUNT(*) FROM {_MAIN_TABLE} WHERE archived = 0 GROUP BY memory_type"
                    ).fetchall()
                    ldist = {r[0]: r[1] for r in lrows}
                    layer_stats = {
                        "self": ldist.get("self_identity", 0),
                        "anchor": ldist.get("mental_model", 0) + ldist.get("anchor", 0),
                        "knowledge": ldist.get("insight", 0) + ldist.get("knowledge", 0),
                        "experience": ldist.get("experience", 0),
                        "context": ldist.get("observation", 0),
                        "total_active": sum(ldist.values()) or 1,
                        "layer_percentages": {k: round(v/(sum(ldist.values()) or 1), 3) for k, v in {
                            "self": ldist.get("self_identity", 0),
                            "anchor": ldist.get("mental_model", 0) + ldist.get("anchor", 0),
                            "knowledge": ldist.get("insight", 0) + ldist.get("knowledge", 0),
                            "experience": ldist.get("experience", 0),
                            "context": ldist.get("observation", 0),
                        }.items()},
                    }
                except Exception:
                    layer_stats = {}
                return {
                    "total_memories": total, "active_memories": hit_active,
                    "reliability": reliability, "hit_distribution": hit_dist,
                    "importance_distribution": imp_dist,
                    "archival_ratio": round(archival_ratio, 3),
                    "graph_snr": round(graph_snr, 3),
                    "layer_distribution": layer_stats,
                    "suggestions": suggestions,
                }
            except Exception as e:
                logger.error("cognition_report failed: %s", e)
                return {"error": str(e), "reliability": "watch"}
