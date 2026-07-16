"""SQLite-backed memory store with FTS5 full-text search and sqlite-vec vector storage.

Schema:
  memories          — core fact table (id, content, content_jieba, agent_space, created_at, updated_at)
  memories_fts      — FTS5 virtual table over content_jieba (Chinese-aware via jieba)
  vec_memories      — sqlite-vec virtual table storing embedding vectors (dim=1024 float32)

Agent spaces:
  Each memory belongs to an ``agent_space`` (string). Multiple agents can
  share the same space (shared memory) or use different spaces (isolated).
  Default space is ``"default"``.
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

# sqlite-vec virtual table name
_VEC_TABLE = "vec_memories"
_FTS_TABLE = "memories_fts"
_MAIN_TABLE = "memories"

_SCHEMA_SQL = f"""
-- Core memory store
CREATE TABLE IF NOT EXISTS {_MAIN_TABLE} (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content       TEXT NOT NULL,
    content_jieba TEXT NOT NULL DEFAULT '',
    agent_space   TEXT NOT NULL DEFAULT 'default',
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- FTS5 full-text index (content-holding: external content table)
CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE}
    USING fts5(
        content_jieba,
        content UNINDEXED,
        agent_space UNINDEXED,
        content={_MAIN_TABLE},
        content_rowid=id,
        tokenize='unicode61'
    );

-- Triggers to keep FTS5 in sync
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

-- Index for agent_space filtering
CREATE INDEX IF NOT EXISTS idx_memories_space ON {_MAIN_TABLE}(agent_space);
CREATE INDEX IF NOT EXISTS idx_memories_created ON {_MAIN_TABLE}(created_at DESC);
"""


def _tokenize(text: str) -> str:
    """Tokenize text with jieba for FTS5 indexing.

    Returns space-separated tokens. Chinese phrases are segmented by jieba;
    English/numeric tokens are preserved as-is.

    Example:
      '用户喜欢Python编程' -> '用户 喜欢 Python 编程'
    """
    if not text:
        return ""
    words = jieba.lcut(text.strip())
    return " ".join(words)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class HybridMemoryStore:
    """Thread-safe SQLite store with FTS5 + vec indexes."""

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
        """Open connection, load extensions, create schema."""
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Create main + FTS tables
        self._conn.executescript(_SCHEMA_SQL)

        # Create vec virtual table
        self._conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS {_VEC_TABLE}
                USING vec0(
                    memory_id INTEGER PRIMARY KEY,
                    embedding float[{self._embedding_dim}]
                )"""
        )

        self._conn.commit()

    # -- Write operations ---------------------------------------------------

    def add_memory(
        self,
        content: str,
        agent_space: str = "default",
        embedding: list[float] | None = None,
    ) -> int | None:
        """Insert a memory and optionally its embedding vector.

        Automatically generates jieba-tokenized version for FTS5 search.
        Returns the new memory ID, or None on failure.
        """
        if not content or not content.strip():
            return None
        content_jieba = _tokenize(content)
        with self._lock:
            try:
                cur = self._conn.execute(
                    f"INSERT INTO {_MAIN_TABLE}(content, content_jieba, agent_space) "
                    f"VALUES (?, ?, ?)",
                    (content.strip(), content_jieba, agent_space),
                )
                memory_id = cur.lastrowid
                if memory_id and embedding is not None:
                    embedding_json = json.dumps(embedding)
                    try:
                        self._conn.execute(
                            f"INSERT INTO {_VEC_TABLE}(memory_id, embedding) VALUES (?, ?)",
                            (memory_id, embedding_json),
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to insert vec for memory %d: %s", memory_id, e
                        )
                self._conn.commit()
                return memory_id
            except Exception as e:
                logger.error("Failed to add memory: %s", e)
                self._conn.rollback()
                return None

    def update_memory(
        self,
        memory_id: int,
        content: str,
        embedding: list[float] | None = None,
    ) -> bool:
        """Update a memory's content and optionally its embedding vector.

        The FTS5 trigger handles re-indexing automatically.
        """
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
                    embedding_json = json.dumps(embedding)
                    try:
                        self._conn.execute(
                            f"INSERT OR REPLACE INTO {_VEC_TABLE}(memory_id, embedding) "
                            f"VALUES (?, ?)",
                            (memory_id, embedding_json),
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to update vec for memory %d: %s", memory_id, e
                        )
                self._conn.commit()
                return True
            except Exception as e:
                logger.error("Failed to update memory %d: %s", memory_id, e)
                self._conn.rollback()
                return False

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID. FTS trigger cleans up FTS index automatically."""
        with self._lock:
            try:
                # Delete from vec table first
                self._conn.execute(
                    f"DELETE FROM {_VEC_TABLE} WHERE memory_id = ?",
                    (memory_id,),
                )
                self._conn.execute(
                    f"DELETE FROM {_MAIN_TABLE} WHERE id = ?",
                    (memory_id,),
                )
                self._conn.commit()
                return self._conn.total_changes > 0
            except Exception as e:
                logger.error("Failed to delete memory %d: %s", memory_id, e)
                self._conn.rollback()
                return False

    # -- Read operations ----------------------------------------------------

    def list_memories(
        self,
        agent_space: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List memories, optionally filtered by agent_space."""
        with self._lock:
            try:
                if agent_space:
                    rows = self._conn.execute(
                        f"SELECT id, content, agent_space, created_at, updated_at "
                        f"FROM {_MAIN_TABLE} WHERE agent_space = ? "
                        f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (agent_space, limit, offset),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT id, content, agent_space, created_at, updated_at "
                        f"FROM {_MAIN_TABLE} "
                        f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    ).fetchall()
                return [self._row_to_dict(r) for r in rows]
            except Exception as e:
                logger.error("Failed to list memories: %s", e)
                return []

    def search_fts(
        self,
        query: str,
        agent_space: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid FTS + LIKE search.

        Strategy:
        1. FTS5 OR query on jieba-tokenized column (best for English)
        2. LIKE fallback on raw content (most reliable for Chinese)
        3. Token-split LIKE on individual key tokens

        Returns memories with FTS5 rank (lower = better), or neutral rank for LIKE.
        """
        if not query or not query.strip():
            return []

        tokenized = _tokenize(query)
        if not tokenized.strip():
            tokenized = query.strip()

        # Build FTS5 OR query
        fts_parts = []
        for t in tokenized.split():
            t = t.strip()
            if t:
                fts_parts.append(f'"{t}"*')
        fts_query = " OR ".join(fts_parts) if fts_parts else query

        with self._lock:
            # Try FTS5 first
            try:
                if agent_space:
                    rows = self._conn.execute(
                        f"SELECT m.id, m.content, m.agent_space, m.created_at, "
                        f"  m.updated_at, rank "
                        f"FROM {_FTS_TABLE} f "
                        f"JOIN {_MAIN_TABLE} m ON f.rowid = m.id "
                        f"WHERE {_FTS_TABLE} MATCH ? AND m.agent_space = ? "
                        f"ORDER BY rank LIMIT ?",
                        (fts_query, agent_space, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT m.id, m.content, m.agent_space, m.created_at, "
                        f"  m.updated_at, rank "
                        f"FROM {_FTS_TABLE} f "
                        f"JOIN {_MAIN_TABLE} m ON f.rowid = m.id "
                        f"WHERE {_FTS_TABLE} MATCH ? "
                        f"ORDER BY rank LIMIT ?",
                        (fts_query, limit),
                    ).fetchall()
                if rows:
                    results = []
                    for r in rows:
                        d = self._row_to_dict(r[:5])
                        d["fts_rank"] = r[5]
                        results.append(d)
                    return results
            except Exception as e:
                logger.debug("FTS5 query failed: %s (falling back to LIKE)", e)

            # Fallback: LIKE search
            try:
                like = f"%{query.strip()}%"
                if agent_space:
                    rows = self._conn.execute(
                        f"SELECT id, content, agent_space, created_at, updated_at "
                        f"FROM {_MAIN_TABLE} "
                        f"WHERE content LIKE ? AND agent_space = ? "
                        f"ORDER BY created_at DESC LIMIT ?",
                        (like, agent_space, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT id, content, agent_space, created_at, updated_at "
                        f"FROM {_MAIN_TABLE} "
                        f"WHERE content LIKE ? "
                        f"ORDER BY created_at DESC LIMIT ?",
                        (like, limit),
                    ).fetchall()

                # Token-split LIKE
                if not rows:
                    # Try each key token separately
                    key_tokens = [t for t in tokenized.split() if len(t) > 1]
                    for token in key_tokens[:5]:
                        like = f"%{token}%"
                        if agent_space:
                            r = self._conn.execute(
                                f"SELECT id, content, agent_space, created_at, updated_at "
                                f"FROM {_MAIN_TABLE} "
                                f"WHERE content LIKE ? AND agent_space = ? "
                                f"ORDER BY created_at DESC LIMIT ?",
                                (like, agent_space, limit),
                            ).fetchall()
                        else:
                            r = self._conn.execute(
                                f"SELECT id, content, agent_space, created_at, updated_at "
                                f"FROM {_MAIN_TABLE} "
                                f"WHERE content LIKE ? "
                                f"ORDER BY created_at DESC LIMIT ?",
                                (like, limit),
                            ).fetchall()
                        rows.extend(r)
                        if len(rows) >= limit:
                            break
                    # Also try bigram split for multi-char Chinese tokens
                    if not rows:
                        for t in key_tokens[:3]:
                            if len(t) > 2:
                                for i in range(len(t) - 1):
                                    bigram = t[i:i+2]
                                    like = f"%{bigram}%"
                                    if agent_space:
                                        r = self._conn.execute(
                                            f"SELECT id, content, agent_space, created_at, updated_at "
                                            f"FROM {_MAIN_TABLE} "
                                            f"WHERE content LIKE ? AND agent_space = ? "
                                            f"ORDER BY created_at DESC LIMIT ?",
                                            (like, agent_space, limit),
                                        ).fetchall()
                                    else:
                                        r = self._conn.execute(
                                            f"SELECT id, content, agent_space, created_at, updated_at "
                                            f"FROM {_MAIN_TABLE} "
                                            f"WHERE content LIKE ? "
                                            f"ORDER BY created_at DESC LIMIT ?",
                                            (like, limit),
                                        ).fetchall()
                                    rows.extend(r)
                                    if len(rows) >= limit * 2:
                                        break

                    # Deduplicate
                    seen = set()
                    deduped = []
                    for r in rows:
                        if r[0] not in seen:
                            seen.add(r[0])
                            deduped.append(r)
                    rows = deduped[:limit]

                results = []
                for r in rows:
                    d = self._row_to_dict(r[:5])
                    d["fts_rank"] = -1.0  # neutral
                    results.append(d)
                return results
            except Exception as e:
                logger.debug("LIKE fallback failed: %s", e)
                return []

    def search_vector(
        self,
        embedding: list[float],
        agent_space: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Vector similarity search via sqlite-vec.

        Returns memories ordered by L2 distance.
        """
        if not embedding:
            return []
        embedding_json = json.dumps(embedding)
        with self._lock:
            try:
                if agent_space:
                    rows = self._conn.execute(
                        f"SELECT m.id, m.content, m.agent_space, m.created_at, "
                        f"  m.updated_at, v.distance "
                        f"FROM {_VEC_TABLE} v "
                        f"JOIN {_MAIN_TABLE} m ON v.memory_id = m.id "
                        f"WHERE v.embedding MATCH ? AND m.agent_space = ? "
                        f"ORDER BY v.distance LIMIT ?",
                        (embedding_json, agent_space, limit),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"SELECT m.id, m.content, m.agent_space, m.created_at, "
                        f"  m.updated_at, v.distance "
                        f"FROM {_VEC_TABLE} v "
                        f"JOIN {_MAIN_TABLE} m ON v.memory_id = m.id "
                        f"WHERE v.embedding MATCH ? "
                        f"ORDER BY v.distance LIMIT ?",
                        (embedding_json, limit),
                    ).fetchall()
                results = []
                for r in rows.fetchall():
                    d = self._row_to_dict(r[:5])
                    d["vec_distance"] = float(r[5])
                    d["vec_similarity"] = 1.0 / (1.0 + float(r[5]))
                    results.append(d)
                return results
            except Exception as e:
                logger.debug("Vector search failed: %s", e)
                return []

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        """Get a single memory by ID."""
        with self._lock:
            try:
                row = self._conn.execute(
                    f"SELECT id, content, agent_space, created_at, updated_at "
                    f"FROM {_MAIN_TABLE} WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                return self._row_to_dict(row) if row else None
            except Exception as e:
                logger.error("Failed to get memory %d: %s", memory_id, e)
                return None

    def count_memories(self, agent_space: str | None = None) -> int:
        """Count memories, optionally filtered by agent_space."""
        with self._lock:
            try:
                if agent_space:
                    row = self._conn.execute(
                        f"SELECT COUNT(*) FROM {_MAIN_TABLE} WHERE agent_space = ?",
                        (agent_space,),
                    ).fetchone()
                else:
                    row = self._conn.execute(
                        f"SELECT COUNT(*) FROM {_MAIN_TABLE}"
                    ).fetchone()
                return row[0] if row else 0
            except Exception:
                return 0

    # -- Utils ---------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: tuple) -> dict[str, Any]:
        return {
            "id": row[0],
            "content": row[1],
            "agent_space": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None