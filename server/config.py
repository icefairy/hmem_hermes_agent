"""HMEM 配置 — 环境变量驱动。"""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        # 基础
        self.debug: bool = os.environ.get("HMEM_DEBUG", "").lower() in ("1", "true", "yes")
        self.api_key: str = os.environ.get("HMEM_API_KEY", "")
        self.host: str = os.environ.get("HMEM_HOST", "0.0.0.0")
        self.port: int = int(os.environ.get("HMEM_PORT", "8000"))

        # 数据库根目录 — 各 namespace 对应的 db 文件存于此目录下
        self.db_root: str = os.environ.get("HMEM_DATA_DIR", "/data/hmem")
        self.embedding_dim: int = int(os.environ.get("EMBEDDING_DIM", "1024"))

        # 嵌入/重排
        self.embedding_base_url: str = os.environ.get("EMBEDDING_BASE_URL", "")
        self.embedding_api_key: str = os.environ.get("EMBEDDING_API_KEY", "")
        self.embedding_model: str = os.environ.get("EMBEDDING_MODEL", "bge-m3")
        self.rerank_model: str = os.environ.get("RERANK_MODEL", "rerankv2m3")

        # Reflect 引擎
        self.reflect_interval: int = int(os.environ.get("REFLECT_INTERVAL", "3600"))
        self.reflect_min_experiences: int = int(os.environ.get("REFLECT_MIN_EXPERIENCES", "50"))
        self.reflect_model: str = os.environ.get("REFLECT_MODEL", "deepseek-v4-flash")