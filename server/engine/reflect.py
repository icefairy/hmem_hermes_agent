"""Reflect 引擎 — 定期分析经验 → 抽象出心智模型。

Pipeline:
  1. 批量读取最新的 N 条经验（memory_type='experience'）
  2. 用 LLM 聚类分析，找出共同模式
  3. 对每个聚类，生成心智模型
  4. 写入心智模型层（memory_type='mental_model'）
  5. 更新记忆关联（parent_id）
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .store import HybridMemoryStore
from .retriever import HybridRetriever
from .embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

_REFLECT_PROMPT = """你是一个经验分析专家。请分析以下多条经验记录，找出共同模式。

每条经验包含：
- content: 经验内容
- mem_action: 动作类型（如 code_generation, qa, debug 等）
- mem_context: 上下文（如语言、框架等）
- mem_outcome: 结果（成功/失败、反馈等）

请执行以下分析：
1. 将经验分组为 2-5 个聚类，每组代表一个共同模式
2. 对每个聚类，提炼为一条心智模型（精炼、可迁移的认知规律）
3. 每个心智模型给出置信度（0-1）

返回 JSON 格式：
{
  "models": [
    {
      "pattern": "用户在XX场景下偏好XX方案",
      "confidence": 0.85,
      "supporting_indices": [0, 2, 5],
      "actionable_advice": "下次遇到同类问题时优先尝试XX"
    }
  ]
}

经验记录：
{experiences_json}
"""


class ReflectEngine:
    def __init__(
        self,
        store: HybridMemoryStore,
        retriever: HybridRetriever,
        embedding_client: EmbeddingClient | None = None,
        min_experiences: int = 50,
        reflection_interval: int = 3600,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._embedding_client = embedding_client
        self._min_experiences = min_experiences
        self._reflection_interval = reflection_interval
        self._last_reflect_time: float = 0.0

    def should_reflect(self) -> bool:
        """判断是否需要执行一轮反思。"""
        now = time.time()
        if now - self._last_reflect_time < self._reflection_interval:
            return False
        total = self._store.count_memories(memory_type="experience")
        return total >= self._min_experiences

    async def run_once(self) -> dict[str, Any]:
        """执行一轮反思。

        需要外部传入 LLM 聊天完成函数（与 Hermes 解耦，不直接依赖）。
        当前返回占位结果——集成 LLM 后在回调中注入。
        """
        # 批量读取最新经验
        experiences = self._store.list_memories(
            memory_type="experience",
            limit=self._min_experiences,
            offset=0,
        )

        if len(experiences) < self._min_experiences:
            return {"models": [], "status": "skipped", "reason": "not enough experiences"}

        logger.info("Reflect: analyzing %d experiences", len(experiences))

        # 这里 placeholder 演示逻辑：
        # 实际使用时，需注入一个 llm_complete(messages) -> str 回调
        # 调用 _REFLECT_PROMPT.format(experiences_json=json.dumps(experiences, ensure_ascii=False))

        # 占位：简单按 mem_action 分组作为聚类
        clusters: dict[str, list[dict]] = {}
        for exp in experiences:
            action = exp.get("mem_action", "unknown")
            clusters.setdefault(action, []).append(exp)

        models = []
        for action, group in clusters.items():
            if len(group) < 3:
                continue
            model_content = (
                f"模式识别：在「{action}」场景下，系统记录了 {len(group)} 条经验。"
                f"典型内容：{group[0].get('content', '')[:100]}"
            )
            model_id = self._store.add_memory(
                content=model_content,
                embedding=None,
                memory_type="mental_model",
            )
            # 关联子记忆
            for exp in group:
                self._store.add_edge(
                    source_id=exp["id"],
                    target_id=model_id,
                    relation="supporting_evidence",
                )
            models.append({
                "model_id": model_id,
                "pattern": model_content,
                "confidence": min(0.5 + len(group) * 0.02, 0.95),
                "supporting_count": len(group),
            })

        self._last_reflect_time = time.time()
        logger.info("Reflect: generated %d mental models", len(models))
        return {"models": models, "status": "ok"}