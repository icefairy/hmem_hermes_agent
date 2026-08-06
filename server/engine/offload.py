"""上下文卸载引擎（Context Offloading）— 会话级卸载存储。

原理参考 TencentDB-Agent-Memory MemoryCore src/offload/：
- 上下文里的大段工具结果/长文本 → 卸载到 refs 文件（原文 100% 可找回）
- SQLite 只存摘要 + node_id + refs_path，注入上下文时用摘要替代原文
- node_id + depends_on 可生成 Mermaid 任务画布（L2 思想）
- 软删除（refs 移入 .trash）保留增量数据，可随时硬删

HMEM 裁剪版设计：
- 零外部依赖（SQLite + 文件系统），namespace 分目录隔离（对齐现有分库）
- 会话级隔离：offload_root/<namespace>/<session_key>/
- 安全：session_key/node_id 白名单校验（防路径穿越）、内容哈希校验、原子写
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 目录/ID 白名单：防路径穿越
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

# 单条内容上限（字符），防滥用
_MAX_CONTENT_CHARS = 10_000_000

# 摘要长度上限（单行，适合注入上下文）
DEFAULT_SUMMARY_CHARS = 120

_SESSIONS_TABLE = "offload_sessions"
_RECORDS_TABLE = "offload_records"

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {_SESSIONS_TABLE} (
    session_key TEXT PRIMARY KEY,
    meta        TEXT NOT NULL DEFAULT '{{}}',
    deleted     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS {_RECORDS_TABLE} (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key  TEXT NOT NULL,
    node_id      TEXT NOT NULL,
    summary      TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT 'text',
    refs_path    TEXT NOT NULL DEFAULT '',
    meta         TEXT NOT NULL DEFAULT '{{}}',
    content_hash TEXT NOT NULL DEFAULT '',
    deleted      INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT '',
    updated_at   TEXT NOT NULL DEFAULT '',
    UNIQUE(session_key, node_id)
);

CREATE INDEX IF NOT EXISTS idx_offload_records_session
    ON {_RECORDS_TABLE}(session_key, created_at);
"""


def _now() -> str:
    """当前 CST (UTC+8) 时间戳。"""
    utc_now = datetime.now(timezone.utc)
    cst_now = utc_now + timedelta(hours=8)
    return cst_now.strftime("%Y-%m-%d %H:%M:%S")


def _to_one_line(text: str, max_chars: int = DEFAULT_SUMMARY_CHARS) -> str:
    """把任意文本压缩成单行摘要（去换行/压缩空白/截断）。

    截断时保留省略号占位，保证总长 ≤ max_chars（测试契约 <=120）。
    """
    s = re.sub(r"\s+", " ", (text or "").strip())
    if len(s) > max_chars:
        s = s[: max_chars - 1].rstrip() + "…"
    return s


def _content_hash(content: str) -> str:
    """内容指纹，用于找回时的完整性校验。"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _mermaid_escape(label: str) -> str:
    """Mermaid 节点 label 转义（双引号/井号）。"""
    return (
        label.replace("&", "&amp;")
        .replace('"', "#quot;")
        .replace("#", "#35;")
    )


class OffloadStore:
    """线程安全的 offload 存储：SQLite 摘要表 + 文件系统原文。

    构造参数对齐 router：db_path 为 namespace 库文件路径（沿用 HMEM 分库），
    data_root 为数据根目录（offload 数据存 data_root/offload/<namespace>/），
    namespace 用于目录隔离。
    """

    def __init__(self, db_path: str, data_root: str, namespace: str = "default") -> None:
        self._db_path = str(Path(db_path).expanduser().resolve())
        self._offload_root = str(
            (Path(data_root).expanduser().resolve() / "offload" / namespace).resolve()
        )
        self._lock = threading.RLock()  # 可重入：put 锁内会调用 ensure_session
        self._conn: sqlite3.Connection | None = None

    # -- 生命周期 ---------------------------------------------------------

    def initialize(self) -> None:
        Path(self._offload_root).mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # -- 内部工具 ---------------------------------------------------------

    @staticmethod
    def _sanitize(name: str) -> str:
        """校验 session_key / node_id 白名单格式，非法则清洗；清洗后仍非法抛 ValueError。

        router 层对明显越界的输入（../../etc）返回 200 + 清洗结果，因此这里
        优先做替换清洗，而不是直接抛异常。
        """
        if not name:
            raise ValueError("标识符不能为空")
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        cleaned = cleaned[:128]
        if not cleaned:
            raise ValueError(f"非法标识符: {name!r}")
        return cleaned

    def _session_dir(self, session_key: str) -> Path:
        return Path(self._offload_root) / session_key

    def _refs_dir(self, session_key: str) -> Path:
        return self._session_dir(session_key) / "refs"

    def _trash_dir(self) -> Path:
        return Path(self._offload_root) / ".trash"

    def _resolve_ref(self, session_key: str, rel_path: str) -> Path:
        """把相对 session 目录的 refs_path 解析为绝对路径，并校验不越界。

        目录结构: offload_root/<session_key>/<rel_path>，防止不同会话同名 node_id 互相覆盖。
        """
        base = (Path(self._offload_root) / session_key).resolve()
        p = (base / rel_path).resolve()
        if not str(p).startswith(str(base) + os.sep):
            raise ValueError(f"refs_path 越界: {rel_path!r}")
        return p

    def _next_node_id(self, session_key: str) -> str:
        """自动生成会话内自增 node_id（node_1, node_2, ...）。"""
        assert self._conn is not None
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM {_RECORDS_TABLE} "
            f"WHERE session_key = ? AND deleted = 0",
            (session_key,),
        ).fetchone()
        seq = (row[0] if row else 0) + 1
        while True:
            nid = f"node_{seq}"
            exists = self._conn.execute(
                f"SELECT 1 FROM {_RECORDS_TABLE} "
                f"WHERE session_key = ? AND node_id = ?",
                (session_key, nid),
            ).fetchone()
            if not exists:
                return nid
            seq += 1

    # -- 写操作 -----------------------------------------------------------

    def ensure_session(self, session_key: str, meta: dict | None = None) -> dict:
        """创建/获取会话卸载空间（幂等）。返回 session 信息。"""
        session_key = self._sanitize(session_key)
        ts = _now()
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                f"INSERT OR IGNORE INTO {_SESSIONS_TABLE} "
                f"(session_key, meta, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_key, json.dumps(meta or {}, ensure_ascii=False), ts, ts),
            )
            # 若之前软删过，重新激活
            self._conn.execute(
                f"UPDATE {_SESSIONS_TABLE} SET deleted = 0, updated_at = ? "
                f"WHERE session_key = ?",
                (ts, session_key),
            )
            self._conn.commit()

        info = self.session_info(session_key)
        assert info is not None
        info["refs_dir"] = str(self._refs_dir(session_key))
        Path(info["refs_dir"]).mkdir(parents=True, exist_ok=True)
        return info

    def session_info(self, session_key: str) -> dict | None:
        """返回会话元信息（含未删除记录数）。"""
        session_key = self._sanitize(session_key)
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute(
                f"SELECT session_key, meta, deleted, created_at, updated_at "
                f"FROM {_SESSIONS_TABLE} WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if not row:
                return None
            cnt = self._conn.execute(
                f"SELECT COUNT(*) FROM {_RECORDS_TABLE} "
                f"WHERE session_key = ? AND deleted = 0",
                (session_key,),
            ).fetchone()[0]
            return {
                "session_key": row[0],
                "meta": json.loads(row[1] or "{}"),
                "deleted": bool(row[2]),
                "created_at": row[3],
                "updated_at": row[4],
                "record_count": cnt,
            }

    def put(
        self,
        session_key: str,
        node_id: str | None,
        content: str,
        summary: str | None = None,
        content_type: str = "text",
        meta: dict | None = None,
    ) -> dict:
        """卸载一条内容：原文写 refs 文件（纯原文，无头），摘要 upsert 进 SQLite。

        node_id 缺省时自动生成 node_{seq}；同一 (session_key, node_id) 重复 put 为覆盖。
        返回该条记录的摘要信息（含 refs_path，供找回）。
        """
        session_key = self._sanitize(session_key)
        if not content or not content.strip():
            raise ValueError("content 不能为空")
        if len(content) > _MAX_CONTENT_CHARS:
            raise ValueError(f"content 超过上限 {_MAX_CONTENT_CHARS} 字符")

        meta = meta or {}
        ts = _now()

        with self._lock:
            assert self._conn is not None
            # 会话必须存在（先 ensure，防孤儿记录）
            sess = self._conn.execute(
                f"SELECT 1 FROM {_SESSIONS_TABLE} WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if not sess:
                self.ensure_session(session_key)

            if node_id:
                node_id = self._sanitize(node_id)
            else:
                node_id = self._next_node_id(session_key)

            # 1) 原文写 refs 文件（纯内容；原子：tmp + rename）
            rel_path = f"refs/{node_id}.md"
            abs_path = self._resolve_ref(session_key, rel_path)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            digest = _content_hash(content)
            tmp_path = abs_path.with_suffix(".md.tmp")
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(tmp_path, abs_path)

            # 2) 摘要 upsert 进 SQLite
            final_summary = _to_one_line(summary) if summary is not None else _to_one_line(content)
            self._conn.execute(
                f"INSERT INTO {_RECORDS_TABLE} "
                f"(session_key, node_id, summary, content_type, refs_path, meta, content_hash, "
                f" created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                f"ON CONFLICT(session_key, node_id) DO UPDATE SET "
                f" summary = excluded.summary, content_type = excluded.content_type, "
                f" refs_path = excluded.refs_path, meta = excluded.meta, "
                f" content_hash = excluded.content_hash, deleted = 0, updated_at = excluded.updated_at",
                (session_key, node_id, final_summary, content_type or "text",
                 rel_path, json.dumps(meta, ensure_ascii=False), digest, ts, ts),
            )
            self._conn.commit()

        logger.info(
            "offload put: session=%s node=%s type=%s summary_len=%d",
            session_key, node_id, content_type, len(final_summary),
        )
        return {
            "session_key": session_key,
            "node_id": node_id,
            "summary": final_summary,
            "content_type": content_type or "text",
            "refs_path": rel_path,
            "content_hash": digest,
            "created_at": ts,
        }

    # -- 读操作 -----------------------------------------------------------

    def _record_to_dict(self, row: tuple) -> dict:
        return {
            "id": row[0],
            "session_key": row[1],
            "node_id": row[2],
            "summary": row[3],
            "content_type": row[4],
            "refs_path": row[5],
            "meta": json.loads(row[6] or "{}"),
            "content_hash": row[7],
            "deleted": bool(row[8]),
            "created_at": row[9],
            "updated_at": row[10],
        }

    def get(self, session_key: str, node_id: str) -> dict | None:
        """按 node_id 找回完整原文（含哈希校验）。找不到/已软删返回 None。"""
        session_key = self._sanitize(session_key)
        node_id = self._sanitize(node_id)
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute(
                f"SELECT id, session_key, node_id, summary, content_type, refs_path, "
                f"  meta, content_hash, deleted, created_at, updated_at "
                f"FROM {_RECORDS_TABLE} "
                f"WHERE session_key = ? AND node_id = ? AND deleted = 0",
                (session_key, node_id),
            ).fetchone()
            if not row:
                return None
            record = self._record_to_dict(row)
            try:
                abs_path = self._resolve_ref(record["session_key"], record["refs_path"])
                content = abs_path.read_text(encoding="utf-8")
            except (OSError, ValueError) as e:
                logger.warning("offload get: 原文读取失败 %s: %s", record["refs_path"], e)
                return None
            if _content_hash(content) != record["content_hash"]:
                logger.warning("offload get: 原文哈希不匹配（可能被篡改） node=%s", node_id)
            return {**record, "content": content}

    def session_index(
        self,
        session_key: str,
        include_deleted: bool = False,
        limit: int = 10_000,
        offset: int = 0,
    ) -> dict:
        """会话索引：全部摘要 + node_id 列表（不含原文，适合注入上下文）。"""
        session_key = self._sanitize(session_key)
        deleted_clause = "" if include_deleted else "AND deleted = 0"
        with self._lock:
            assert self._conn is not None
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM {_RECORDS_TABLE} "
                f"WHERE session_key = ? {deleted_clause}",
                (session_key,),
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT id, session_key, node_id, summary, content_type, refs_path, "
                f"  meta, content_hash, deleted, created_at, updated_at "
                f"FROM {_RECORDS_TABLE} "
                f"WHERE session_key = ? {deleted_clause} "
                f"ORDER BY id ASC LIMIT ? OFFSET ?",
                (session_key, limit, offset),
            ).fetchall()
            return {
                "session_key": session_key,
                "count": total,
                "limit": limit,
                "offset": offset,
                "records": [self._record_to_dict(r) for r in rows],
            }

    def canvas_mermaid(self, session_key: str) -> str:
        """生成 Mermaid 任务画布：节点=动作，边=依赖（meta.depends_on），click 可下钻。"""
        session_key = self._sanitize(session_key)
        index = self.session_index(session_key, limit=10_000)
        records = index["records"]
        if not records:
            return "flowchart TD\n    empty[\"（empty session）\"]\n"

        lines = ["flowchart TD"]
        node_ids: set[str] = set()
        for r in records:
            node_ids.add(r["node_id"])
            label = _mermaid_escape(_to_one_line(r["summary"], 60) or r["node_id"])
            lines.append(f'    {r["node_id"]}["{label}"]')

        for r in records:
            deps = r["meta"].get("depends_on") or []
            if isinstance(deps, str):
                deps = [deps]
            for d in deps:
                if d in node_ids:
                    lines.append(f"    {d} --> {r['node_id']}")

        for r in records:
            lines.append(
                f'    click {r["node_id"]} "refs/{r["node_id"]}.md" "查看原文"'
            )
        return "\n".join(lines) + "\n"

    # -- 删除 ---------------------------------------------------------------

    def delete_session(self, session_key: str, hard: bool = False) -> dict:
        """删除会话。

        - soft（默认）：deleted=1，refs 目录移入 .trash，保留增量数据
        - hard：物理删除记录和原文文件
        返回 {"mode": "soft"|"hard", "count": N}。
        """
        session_key = self._sanitize(session_key)
        ts = _now()
        with self._lock:
            assert self._conn is not None
            sess = self._conn.execute(
                f"SELECT 1 FROM {_SESSIONS_TABLE} WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            if not sess:
                return {"mode": "none", "count": 0}

            if hard:
                cur = self._conn.execute(
                    f"DELETE FROM {_RECORDS_TABLE} WHERE session_key = ?",
                    (session_key,),
                )
                count = cur.rowcount
                self._conn.execute(
                    f"DELETE FROM {_SESSIONS_TABLE} WHERE session_key = ?",
                    (session_key,),
                )
                self._conn.commit()
                shutil.rmtree(self._session_dir(session_key), ignore_errors=True)
                logger.info("offload 硬删会话: %s", session_key)
                return {"mode": "hard", "count": count}
            else:
                cur = self._conn.execute(
                    f"UPDATE {_RECORDS_TABLE} SET deleted = 1, updated_at = ? "
                    f"WHERE session_key = ?",
                    (ts, session_key),
                )
                count = cur.rowcount
                self._conn.execute(
                    f"UPDATE {_SESSIONS_TABLE} SET deleted = 1, updated_at = ? "
                    f"WHERE session_key = ?",
                    (ts, session_key),
                )
                self._conn.commit()
                # refs 移入 .trash（保留增量数据）
                session_dir = self._session_dir(session_key)
                if session_dir.is_dir():
                    trash = self._trash_dir()
                    trash.mkdir(parents=True, exist_ok=True)
                    target = trash / f"{session_key}_{ts.replace(' ', '_').replace(':', '-')}"
                    shutil.move(str(session_dir), str(target))
                logger.info("offload 软删会话: %s", session_key)
                return {"mode": "soft", "count": count}
