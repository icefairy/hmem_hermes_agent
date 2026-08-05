"""数据库定时备份路由。

功能：
- 每日自动备份所有 namespace 的数据库（gzip 压缩）
- 最多保留 30 天备份
- 支持手动触发、列出、删除单个备份
"""

from __future__ import annotations

import asyncio
import datetime
import gzip
import glob
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, HTTPException

from config import Settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["backup"])

# ── 备份目录 ──────────────────────────────────────────────
# 默认存储在 db_root 下的 backups 子目录（容器内 /data/hmem/backups）
# 宿主机通过 bind mount 持久化到 /root/codes/hmem/server/data/backups


def _backup_dir(db_root: str) -> str:
    return os.path.join(db_root, "backups")


def _ensure_backup_dir(db_root: str) -> str:
    d = _backup_dir(db_root)
    os.makedirs(d, exist_ok=True)
    return d


# ── 核心备份逻辑 ──────────────────────────────────────────
def do_backup(db_root: str) -> list[dict[str, Any]]:
    """对 db_root 下所有 *.db 文件做一次全量备份，返回备份文件列表。"""
    backup_dir = _ensure_backup_dir(db_root)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    namespace_ts = datetime.datetime.now().strftime("%Y-%m-%d")

    # 收集所有 .db 文件
    db_files = sorted(glob.glob(os.path.join(db_root, "*.db")))
    if not db_files:
        logger.warning("No .db files found in %s", db_root)
        return []

    created = []
    for db_path in db_files:
        ns = Path(db_path).stem  # e.g. "default"
        db_size = os.path.getsize(db_path)
        gz_path = os.path.join(backup_dir, f"{ns}_{ts}.db.gz")

        # 用 gzip 压缩
        with open(db_path, "rb") as f_in:
            with gzip.open(gz_path, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)

        gz_size = os.path.getsize(gz_path)
        ratio = (1 - gz_size / max(db_size, 1)) * 100

        created.append({
            "namespace": ns,
            "filename": f"{ns}_{ts}.db.gz",
            "original_size": db_size,
            "compressed_size": gz_size,
            "compression_ratio": round(ratio, 1),
            "created_at": namespace_ts,
        })
        logger.info(
            "Backup %s (%.1f KB → %.1f KB, %.1f%%)",
            Path(db_path).name, db_size / 1024, gz_size / 1024, ratio,
        )

    # 删除超过 30 天的旧备份
    _prune_old_backups(backup_dir)

    return created


def _prune_old_backups(backup_dir: str, max_days: int = 30) -> int:
    """删除超过 max_days 天的备份文件，返回删除数量。"""
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=max_days)
    deleted = 0

    for f in glob.glob(os.path.join(backup_dir, "*.db.gz")):
        mtime = os.path.getmtime(f)
        file_date = datetime.datetime.fromtimestamp(mtime)
        if file_date < cutoff:
            os.remove(f)
            deleted += 1
            logger.info("Pruned old backup: %s (mtime=%s)", os.path.basename(f), file_date)

    if deleted > 0:
        logger.info("Pruned %d backup file(s) older than %d days", deleted, max_days)

    return deleted


def list_backups(db_root: str) -> list[dict[str, Any]]:
    """列出所有备份文件信息。"""
    backup_dir = _backup_dir(db_root)
    if not os.path.isdir(backup_dir):
        return []

    backups = []
    for f in sorted(glob.glob(os.path.join(backup_dir, "*.db.gz"))):
        name = os.path.basename(f)
        size = os.path.getsize(f)
        mtime = os.path.getmtime(f)
        # 解析 namespace 和日期
        parts = name.replace(".db.gz", "").split("_")
        ns = parts[0] if parts else "?"
        file_date = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        backups.append({
            "namespace": ns,
            "filename": name,
            "size_bytes": size,
            "size_kb": round(size / 1024, 1),
            "created_at": file_date,
        })

    return backups


# ── API 路由 ──────────────────────────────────────────────

@router.post("/backup", response_model=None)
async def trigger_backup(req: Request):
    """手动触发一次全量备份。"""
    settings: Settings = req.app.state.settings
    try:
        created = do_backup(settings.db_root)
        return {
            "status": "ok",
            "created": len(created),
            "backups": created,
            "message": f"备份完成，共备份 {len(created)} 个数据库",
        }
    except Exception as e:
        logger.error("Backup failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Backup failed: {e}")


@router.get("/backup")
async def list_all_backups(req: Request):
    """列出所有备份。"""
    settings: Settings = req.app.state.settings
    backups = list_backups(settings.db_root)
    return {
        "count": len(backups),
        "backups": backups,
        "backup_dir": _backup_dir(settings.db_root),
    }


@router.delete("/backup/{filename}")
async def delete_backup(req: Request, filename: str):
    """删除指定的备份文件。"""
    settings: Settings = req.app.state.settings
    backup_dir = _backup_dir(settings.db_root)
    file_path = os.path.join(backup_dir, filename)

    if not os.path.isfile(file_path):
        raise HTTPException(404, f"Backup file not found: {filename}")

    os.remove(file_path)
    return {"status": "ok", "deleted": filename}


@router.get("/backup/stats")
async def backup_stats(req: Request):
    """获取备份统计信息（最新备份时间、保留策略等）。"""
    settings: Settings = req.app.state.settings
    backups = list_backups(settings.db_root)

    latest = None
    if backups:
        latest_file = backups[-1]
        latest = {
            "filename": latest_file["filename"],
            "created_at": latest_file["created_at"],
            "size_kb": latest_file["size_kb"],
        }

    total_size_kb = round(sum(b["size_bytes"] for b in backups) / 1024, 1)

    return {
        "total_count": len(backups),
        "total_size_kb": total_size_kb,
        "retention_days": 30,
        "latest_backup": latest,
        "backup_dir": _backup_dir(settings.db_root),
    }


# ── 后台定时任务 ──────────────────────────────────────────
# 每个检查周期（默认 1 小时）检查是否需要执行每日备份
_BACKUP_CHECK_INTERVAL = 3600  # 秒
_LAST_BACKUP_DATE: dict[str, str] = {}  # db_root -> last date string (YYYY-MM-DD)


async def _backup_scheduler(app: FastAPI) -> None:
    """后台定时器：每天自动备份一次。"""
    settings = app.state.settings
    db_root = settings.db_root

    # 启动时如果今天还没备份过，立即执行
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    last = _LAST_BACKUP_DATE.get(db_root)
    if last != today:
        logger.info("Running initial daily backup at startup")
        await _safe_backup(db_root)
        _LAST_BACKUP_DATE[db_root] = today

    while True:
        await asyncio.sleep(_BACKUP_CHECK_INTERVAL)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        last = _LAST_BACKUP_DATE.get(db_root)
        if last != today:
            logger.info("Running scheduled daily backup")
            await _safe_backup(db_root)
            _LAST_BACKUP_DATE[db_root] = today


async def _safe_backup(db_root: str) -> None:
    """安全执行备份（异常不影响主进程）。"""
    try:
        created = do_backup(db_root)
        logger.info("Daily backup completed: %d database(s) backed up", len(created))
    except Exception as e:
        logger.error("Daily backup failed (will retry next cycle): %s", e, exc_info=True)
