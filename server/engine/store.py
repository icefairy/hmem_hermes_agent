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
from contextlib import suppress
from datetime import datetime, timezone
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

VALID_MEMORY_TYPES = {"observation", "experience", "insight", "mental_model", "knowledge"}

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


def _tokenize(text: str) -> str:
    if not text:
        return ""
    words = jieba.lcut(text.strip())
    return " ".join(words)


def _now() -> str:
    """Returns current CST (UTC+8) formatted timestamp."""
    from datetime import timedelta

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
    ) -> int | None:
        if not content or not content.strip():
            return None
        # Validate memory_type
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = "experience"
        content_jieba = _tokenize(content)
        # Python-side timestamp to avoid SQLite strftime %% issues
        ts = created_at or _now()
        with self._lock:
            try:
                cur = self._conn.execute(
                    f"INSERT INTO {_MAIN_TABLE} "
                    f"(content, content_jieba, memory_type, "
                    f" mem_action, mem_context, mem_outcome, mem_metadata, parent_id, created_at, updated_at, "
                    f" doc_id, doc_uri, doc_title, chunk_index, doc_category, doc_tags) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def delete_memory(self, memory_id: int) -> bool:
        with self._lock:
            try:
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
        """级联删除某个文档的所有 chunk（含向量/边/FTS），返回删除条数。"""
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
                    f"  doc_id, doc_uri, doc_title, chunk_index, doc_category, doc_tags "
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
                    f"  doc_id, doc_uri, doc_title, chunk_index, doc_category, doc_tags "
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
        limit: int = 200,
    ) -> dict[str, Any]:
        """返回力导向图数据。"""
        nodes: list[dict] = []
        edges: list[dict] = []
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT id, content, memory_type, "
                    f"  hit_count FROM {_MAIN_TABLE} "
                    f"ORDER BY created_at DESC LIMIT ?",
                    (limit,),
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
        if len(row) > 16:
            if row[16]:
                d["category"] = row[16]
            if row[17]:
                d["tags"] = row[17]
        return d

    def close(self) -> None:
        conn = self._conn  # type: ignore[reportAttributeAccessIssue]  # close 前可能未 initialize，与 __init__ 注释一致
        if conn:
            with suppress(Exception):
                conn.close()
        self._conn = None  # type: ignore[assignment]  # 类注解非 Optional，close 后不再使用
