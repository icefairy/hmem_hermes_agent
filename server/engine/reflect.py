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
from typing import Any, Callable, Coroutine

from engine.store import HybridMemoryStore
from engine.retriever import HybridRetriever
from engine.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

# 异步 LLM 回调签名：接受 messages 列表，返回字符串响应
LlmCompleteFn = Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]]

_REFLECT_PROMPT = """你是一个经验分析专家。请分析以下多条经验记录，找出共同模式。

每条经验包含：
- content: 经验内容
- mem_action: 动作类型（如 code_generation, qa, debug 等）
- mem_context: 上下文（如语言、框架等）

请执行以下分析：
1. 将经验分组为 2-5 个聚类，每组代表一个共同模式
2. 对每个聚类，提炼为一条心智模型（精炼、可迁移的认知规律）
3. 每个心智模型给出置信度（0-1）

返回严格的 JSON 格式（不要 markdown 包裹，只输出 JSON）：
{"models": [{"pattern": "...
", "confidence": 0.85, "supporting_indices": [0, 2, 5], "actionable_advice": "..."}]}

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
        llm_complete: LlmCompleteFn | None = None,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._embedding_client = embedding_client
        self._min_experiences = min_experiences
        self._reflection_interval = reflection_interval
        self._llm_complete = llm_complete
        self._last_reflect_time: float = 0.0

    def should_reflect(self) -> bool:
        now = time.time()
        if now - self._last_reflect_time < self._reflection_interval:
            return False
        total = self._store.count_memories(memory_type="experience")
        return total >= self._min_experiences

    async def run_once(self) -> dict[str, Any]:
        """执行一轮反思。

        如果注入了 llm_complete 回调，使用 LLM 进行真正的聚类分析；
        否则回退到按 mem_action 分组的占位逻辑。
        """
        experiences = self._store.list_memories(
            memory_type="experience",
            limit=self._min_experiences,
            offset=0,
        )

        if len(experiences) < self._min_experiences:
            return {"models": [], "status": "skipped", "reason": "not enough experiences"}

        logger.info("Reflect: analyzing %d experiences", len(experiences))

        # 尝试 LLM 模式识别
        if self._llm_complete:
            try:
                result = await self._reflect_with_llm(experiences)
                if result.get("models"):
                    self._last_reflect_time = time.time()
                    return result
            except Exception as e:
                logger.warning("LLM reflect failed, falling back: %s", e)

        # 占位：按 mem_action 分组
        models = self._fallback_reflect(experiences)
        self._last_reflect_time = time.time()
        return {"models": models, "status": "ok"}

    async def _reflect_with_llm(
        self, experiences: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """使用 LLM 进行真正的聚类分析。"""
        # 精简经验数据（只保留关键字段）
        exps_compact = []
        for i, exp in enumerate(experiences):
            exps_compact.append({
                "index": i,
                "content": exp.get("content", "")[:200],
                "mem_action": exp.get("mem_action", ""),
                "mem_context": exp.get("mem_context", "{}"),
            })

        prompt = _REFLECT_PROMPT.format(
            experiences_json=json.dumps(exps_compact, ensure_ascii=False, indent=2)
        )

        messages = [
            {"role": "system", "content": "你是一个经验分析专家。始终返回严格 JSON。"},
            {"role": "user", "content": prompt},
        ]

        text = await self._llm_complete(messages)

        # 清理可能的 markdown 包裹
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, falling back")
            return {"models": [], "status": "llm_json_error"}

        raw_models = data.get("models", [])
        if not raw_models:
            return {"models": [], "status": "no_models_found"}

        models = []
        for rm in raw_models:
            pattern = rm.get("pattern", "")
            confidence = min(max(rm.get("confidence", 0.5), 0.0), 1.0)
            supporting_indices = rm.get("supporting_indices", [])
            advice = rm.get("actionable_advice", "")

            if not pattern:
                continue

            # 写入心智模型
            full_content = pattern
            if advice:
                full_content += f"\n建议: {advice}"

            model_id = self._store.add_memory(
                content=full_content,
                embedding=None,
                memory_type="mental_model",
                mem_metadata=json.dumps({"confidence": confidence, "actionable_advice": advice}),
            )

            # 创建关联边
            for idx in supporting_indices:
                if 0 <= idx < len(experiences):
                    self._store.add_edge(
                        source_id=experiences[idx]["id"],
                        target_id=model_id,
                        relation="supporting_evidence",
                    )

            models.append({
                "model_id": model_id,
                "pattern": pattern,
                "confidence": confidence,
                "supporting_count": len(supporting_indices),
            })

        return {"models": models, "status": "ok"}

    def _fallback_reflect(
        self, experiences: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """占位：按 mem_action 分组。"""
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

        return models