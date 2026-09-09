"""受保护命名空间/知识库定义 — 统一删除防护的单点来源。

被列入 PROTECTED_NAMESPACES 的库视为系统保留库：
  - 后端删除接口（命名空间 / 知识库）一律拒绝，返回 400
  - 列表接口返回 protected=True，前端据此隐藏删除入口并打“不可删除”标记

当前保护区：
  - default: 记忆默认库，误删会清空全部记忆
  - kb:      手工维护的知识库（xuanji / health / meta 等文档），内容不可重建
"""

from __future__ import annotations

PROTECTED_NAMESPACES: frozenset[str] = frozenset({"default", "kb"})