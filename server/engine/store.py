"""SQLite-backed memory store with FTS5 full-text search and sqlite-vec vector storage.

Schema v2:
  memories           — core fact table (id, content, content_jieba, agent_space, memory_type,
                       mem_action, mem_context, mem_outcome, mem_metadata, parent_id,
                       hit_count, created_at, updated_at)
  memories_fts       — FTS5 virtual table over content_jieba (Chinese-aware via jieba)
  vec_memories       — sqlite-vec virtual table storing embedding vectors (dim=1024 float32)
  memory_edges       — graph edges for knowledge graph / causal chains
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jieba
import sqlite_vec

logger = logging.getLogger(__name__)

_VEC_TABLE = "vec_memories"
_FTS_TABLE = "memories_fts"
_MAIN_TABLE = "memories"
_EDGE_TABLE = "memory_edges"

_SCHEMA_V2_SQL = f"""
CREATE TABLE IF NOT EXISTS {_MAIN_TABLE} (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content       TEXT NOT NULL,
    content_jieba TEXT NOT NULL DEFAULT '',
    agent_space   TEXT NOT NULL DEFAULT 'default',
    memory_type   TEXT NOT NULL DEFAULT 'experience',
    mem_action    TEXT DEFAULT '',
    mem_context   TEXT DEFAULT '{{}}',
    mem_outcome   TEXT DEFAULT '{{}}',
    mem_metadata  TEXT DEFAULT '{{}}',
    parent_id     INTEGER DEFAULT NULL,
    hit_count     INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (strftime('%%Y-%%m-%%dT%%H:%%M:%%fZ', 'now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%%Y-%%m-%%dT%%H:%%M:%%fZ', 'now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE}
    USING fts5(
        content_jieba,
        content UNINDEXED,
        agent_space UNINDEXED,
        content={_MAIN_TABLE},
        content_rowid=id,
        tokenize='unicode61'
    );

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON {_MAIN_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}(rowid, content_jieba, content, agent_space)
        VALUES (new.id, new.content_jieba, new.content, new.agent_space);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON {_MAIN_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, content_jieba, content, agent_space)
        VALUES ('delete', old.id, old.content_jieba, old.content, old.agent_space);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON {_MAIN_TABLE} BEGIN
    INSERT INTO {_FTS_TABLE}({_FTS_TABLE}, rowid, content_jieba, content, agent_space)
        VALUES ('delete', old.id, old.content_jieba, old.content, old.agent_space);
    INSERT INTO {_FTS_TABLE}(rowid, content_jieba, content, agent_space)
        VALUES (new.id, new.content_jieba, new.content, new.agent_space);
END;

CREATE INDEX IF NOT EXISTS idx_memories_space ON {_MAIN_TABLE}(agent_space);
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
    created_at  TEXT NOT NULL DEFAULT (strftime('%%Y-%%m-%%dT%%H:%%M:%%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON {_EDGE_TABLE}(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON {_EDGE_TABLE}(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON {_EDGE_TABLE}(relation);
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


def _tokenize(text: str) -> str:
    if not text:
        return ""
    words = jieba.lcut(text.strip())
    return " ".join(words)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        self._conn: sqlite3.Connection | None = None

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
        try:
            self._conn.executescript(_MIGRATE_V1_TO_V2)
        except Exception:
            pass  # columns already exist

        # Create v2 schema (CREATE IF NOT EXISTS — idempotent)
        self._conn.executescript(_SCHEMA_V2_SQL)

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
        agent_space: str = "default",
        embedding: list[float] | None = None,
        memory_type: str = "experience",
        mem_action: str | None = None,
        mem_context: str | None = None,
        mem_outcome: str | None = None,
        mem_metadata: str | None = None,
        parent_id: int | None = None,
    ) -> int | None:
        if not content or not content.strip():
            return None
        content_jieba = _tokenize(content)
        with self._lock:
            try:
                cur = self._conn.execute(
                    f"INSERT INTO {_MAIN_TABLE} "
                    f"(content, content_jieba, agent_space, memory_type, "
                    f" mem_action, mem_context, mem_outcome, mem_metadata, parent_id) "
                    f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        content.strip(), content_jieba, agent_space, memory_type,
                        mem_action or "", mem_context or "{}", mem_outcome or "{}",
                        mem_metadata or "{}", parent_id,
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
                self._conn.commit()
                return memory_id
            except Exception as e:
                logger.error("add_memory failed: %s", e)
                self._conn.rollback()
                return None

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
                        self._conn.execute(
                            f"INSERT OR REPLACE INTO {_VEC_TABLE}(memory_id, embedding) "
                            f"VALUES (?, ?)",
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
            except Exception:
                pass

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

    # -- Read operations ----------------------------------------------------

    def list_memories(
        self,
        agent_space: str | None = None,
        limit: int = 50,
        offset: int = 0,
        memory_type: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            try:
                where_parts = []
                params: list[Any] = []
                if agent_space:
                    where_parts.append("agent_space = ?")
                    params.append(agent_space)
                if memory_type:
                    where_parts.append("memory_type = ?")
                    params.append(memory_type)
                where = "WHERE " + " AND ".join(where_parts) if where_parts else ""
                rows = self._conn.execute(
                    f"SELECT id, content, content_jieba, agent_space, memory_type, "
                    f"  mem_action, mem_context, mem_outcome, mem_metadata, parent_id, "
                    f"  hit_count, created_at, updated_at "
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
                    f"SELECT id, content, content_jieba, agent_space, memory_type, "
                    f"  mem_action, mem_context, mem_outcome, mem_metadata, parent_id, "
                    f"  hit_count, created_at, updated_at "
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
                    f"SELECT m.id, m.content, m.agent_space, m.memory_type, "
                    f"  m.created_at, m.updated_at "
                    f"FROM {_EDGE_TABLE} e "
                    f"JOIN {_MAIN_TABLE} m ON e.source_id = m.id "
                    f"WHERE e.target_id = ? AND e.relation = 'supporting_evidence' "
                    f"ORDER BY m.created_at DESC LIMIT 100",
                    (parent_id,),
                ).fetchall()
                return [
                    {"id": r[0], "content": r[1], "agent_space": r[2],
                     "memory_type": r[3], "created_at": r[4], "updated_at": r[5]}
                    for r in rows
                ]
            except Exception:
                return []

    def search_fts(
        self,
        query: str,
        agent_space: str | None = None,
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
                if agent_space:
                    rows = self._conn.execute(
                        f"SELECT m.id, m.content, m.agent_space, m.memory_type, "
                        f"  m.created_at, m.updated_at, rank "
                        f"FROM {_FTS_TABLE} f "
                        f"JOIN {_MAIN_TABLE} m ON f.rowid = m.id "
                        f"WHERE {_FTS_TABLE} MATCH ? AND m.agent_space = ? "
                        f"ORDER BY rank LIMIT ?",
                        (fts_query, agent_space, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT m.id, m.content, m.agent_space, m.memory_type, "
                        f"  m.created_at, m.updated_at, rank "
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
                            "id": r[0], "content": r[1], "agent_space": r[2],
                            "memory_type": r[3], "created_at": r[4], "updated_at": r[5],
                        }
                        d["fts_rank"] = r[6]
                        results.append(d)
                    return results
            except Exception as e:
                logger.debug("FTS5 failed: %s", e)

            # LIKE fallback
            try:
                like = f"%{query.strip()}%"
                if agent_space:
                    rows = self._conn.execute(
                        f"SELECT id, content, agent_space, memory_type, "
                        f"  created_at, updated_at "
                        f"FROM {_MAIN_TABLE} "
                        f"WHERE content LIKE ? AND agent_space = ? "
                        f"ORDER BY created_at DESC LIMIT ?",
                        (like, agent_space, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT id, content, agent_space, memory_type, "
                        f"  created_at, updated_at "
                        f"FROM {_MAIN_TABLE} "
                        f"WHERE content LIKE ? "
                        f"ORDER BY created_at DESC LIMIT ?",
                        (like, limit),
                    ).fetchall()

                if not rows:
                    key_tokens = [t for t in tokenized.split() if len(t) > 1]
                    for token in key_tokens[:5]:
                        like = f"%{token}%"
                        if agent_space:
                            r = self._conn.execute(
                                f"SELECT id, content, agent_space, memory_type, "
                                f"  created_at, updated_at "
                                f"FROM {_MAIN_TABLE} "
                                f"WHERE content LIKE ? AND agent_space = ? "
                                f"ORDER BY created_at DESC LIMIT ?",
                                (like, agent_space, limit),
                            ).fetchall()
                        else:
                            r = self._conn.execute(
                                f"SELECT id, content, agent_space, memory_type, "
                                f"  created_at, updated_at "
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
                    {"id": r[0], "content": r[1], "agent_space": r[2],
                     "memory_type": r[3], "created_at": r[4], "updated_at": r[5],
                     "fts_rank": -1.0}
                    for r in rows
                ]
            except Exception as e:
                logger.debug("LIKE fallback failed: %s", e)
                return []

    def search_vector(
        self,
        embedding: list[float],
        agent_space: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not embedding:
            return []
        embedding_json = json.dumps(embedding)
        with self._lock:
            try:
                if agent_space:
                    rows = self._conn.execute(
                        f"SELECT m.id, m.content, m.agent_space, m.memory_type, "
                        f"  m.created_at, m.updated_at, v.distance "
                        f"FROM {_VEC_TABLE} v "
                        f"JOIN {_MAIN_TABLE} m ON v.memory_id = m.id "
                        f"WHERE v.embedding MATCH ? AND m.agent_space = ? "
                        f"ORDER BY v.distance LIMIT ?",
                        (embedding_json, agent_space, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT m.id, m.content, m.agent_space, m.memory_type, "
                        f"  m.created_at, m.updated_at, v.distance "
                        f"FROM {_VEC_TABLE} v "
                        f"JOIN {_MAIN_TABLE} m ON v.memory_id = m.id "
                        f"WHERE v.embedding MATCH ? "
                        f"ORDER BY v.distance LIMIT ?",
                        (embedding_json, limit),
                    ).fetchall()
                results = []
                for r in rows.fetchall():
                    d = {
                        "id": r[0], "content": r[1], "agent_space": r[2],
                        "memory_type": r[3], "created_at": r[4], "updated_at": r[5],
                    }
                    d["vec_distance"] = float(r[6])
                    d["vec_similarity"] = 1.0 / (1.0 + float(r[6]))
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

    def count_by_space(self) -> dict[str, int]:
        with self._lock:
            try:
                rows = self._conn.execute(
                    f"SELECT agent_space, COUNT(*) FROM {_MAIN_TABLE} "
                    f"GROUP BY agent_space"
                ).fetchall()
                return {r[0]: r[1] for r in rows}
            except Exception:
                return {}

    def get_graph(
        self,
        agent_space: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """返回力导向图数据。"""
        nodes: list[dict] = []
        edges: list[dict] = []
        with self._lock:
            try:
                # Get recent memories as nodes
                if agent_space:
                    rows = self._conn.execute(
                        f"SELECT id, content, agent_space, memory_type, "
                        f"  hit_count FROM {_MAIN_TABLE} "
                        f"WHERE agent_space = ? ORDER BY created_at DESC LIMIT ?",
                        (agent_space, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT id, content, agent_space, memory_type, "
                        f"  hit_count FROM {_MAIN_TABLE} "
                        f"ORDER BY created_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()

                id_set = {r[0] for r in rows}
                nodes = [
                    {
                        "id": r[0],
                        "label": r[1][:60] + ("…" if len(r[1]) > 60 else ""),
                        "agent_space": r[2],
                        "type": r[3],
                        "hit_count": r[4],
                    }
                    for r in rows
                ]

                # Get edges that connect visible nodes
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
        return {
            "id": row[0],
            "content": row[1],
            "content_jieba": row[2],
            "agent_space": row[3],
            "memory_type": row[4],
            "mem_action": row[5],
            "mem_context": row[6],
            "mem_outcome": row[7],
            "mem_metadata": row[8],
            "parent_id": row[9],
            "hit_count": row[10],
            "created_at": row[11],
            "updated_at": row[12],
        }

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None