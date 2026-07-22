"""Reflect 引擎 — 三段式知识提炼 Pipeline

Pipeline:
  1. observation → experience: 批量分析原始观察，补全 action/context/outcome 结构化字段
  2. experience → insight: 聚类分析经验，提炼可复用的规律/洞见
  3. insight → mental_model: 跨领域交叉分析，蒸馏为心智模型（世界观/决策框架）

每轮只处理一个阶段，按优先级依次推进：
  - 有足够 observation → 执行 Stage 1
  - 否则有足够 experience → 执行 Stage 2
  - 否则有足够 insight → 执行 Stage 3
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Coroutine

from engine.store import HybridMemoryStore
from engine.retriever import HybridRetriever
from engine.embeddings import EmbeddingClient
from engine.dedup import dedup_before_add

logger = logging.getLogger(__name__)

# 异步 LLM 回调签名：接受 messages 列表，返回字符串响应
LlmCompleteFn = Callable[[list[dict[str, str]]], Coroutine[Any, Any, str]]

# -- Prompt 模板 --------------------------------------------------------------

_PROMPT_OBSERVATION_ENRICH = """你是一个经验分析师。分析以下原始观察记录，补充经验结构。

每条观察只包含 content。
请推断：
- mem_action: 可能的动作类型（如 code_generation, debug, qa, review, research, planning 等）
- mem_context: 上下文摘要（JSON 对象，如 {{"lang": "python", "framework": "fastapi"}}）
- mem_outcome: 结果摘要（JSON 对象，如 {{"result": "success", "quality": "good"}}）
- confidence: 你对以上推断的置信度 (0-1)

返回严格的 JSON 数组格式（不要 markdown 包裹，只输出 JSON）：
[
  {{
    "index": 0,
    "action": "...",
    "context": {{...}},
    "outcome": {{...}},
    "confidence": 0.8
  }}
]

观察记录：
{observation_json}
"""

_PROMPT_EXPERIENCE_INSIGHT = """你是一个经验模式挖掘专家。分析以下经验记录，提炼可复用的洞见。

每条经验包含：
- content: 经验内容
- mem_action: 动作类型
- mem_context: 上下文

请执行以下分析：
1. 将经验分组为 2-5 个聚类，每组代表一个共同模式
2. 对每个聚类，提炼为一条洞见（insight）— 精炼、可迁移的规律
3. 每条洞见给出：
   - pattern: 核心模式描述
   - confidence: 置信度 (0-1)
   - supporting_indices: 支撑该洞见的经验索引列表
   - actionable_advice: 行动建议

返回严格的 JSON 格式（不要 markdown 包裹，只输出 JSON）：
{{"insights": [{{"pattern": "...", "confidence": 0.85, "supporting_indices": [0, 2, 5], "actionable_advice": "..."}}]}}

经验记录：
{experiences_json}
"""

_PROMPT_INSIGHT_MENTAL_MODEL = """你是一个知识蒸馏专家。从以下多条洞见中提炼心智模型。

每条洞见包含：
- pattern: 核心模式
- actionable_advice: 行动建议

请执行：
1. 寻找洞见之间的关联和冲突
2. 合并相关的洞见为更高层次的思维框架
3. 对每个心智模型给出：
   - name: 模型名称（简短有力）
   - principle: 核心原则/世界观
   - applicability: 适用场景
   - counter_indicators: 不适用的情况
   - confidence: 置信度 (0-1)
   - supporting_indices: 支撑的洞见索引

返回严格的 JSON 格式（不要 markdown 包裹，只输出 JSON）：
{{"models": [{{"name": "...", "principle": "...", "applicability": "...", "counter_indicators": "...", "confidence": 0.9, "supporting_indices": [0, 1]}}]}}

洞见记录：
{insights_json}
"""


class ReflectEngine:
    def __init__(
        self,
        store: HybridMemoryStore,
        retriever: HybridRetriever,
        embedding_client: EmbeddingClient | None = None,
        min_experiences: int = 50,
        min_observations: int = 30,
        min_insights: int = 5,
        reflection_interval: int = 3600,
        llm_complete: LlmCompleteFn | None = None,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._embedding_client = embedding_client
        self._min_experiences = min_experiences
        self._min_observations = min_observations
        self._min_insights = min_insights
        self._reflection_interval = reflection_interval
        self._llm_complete = llm_complete
        self._last_reflect_time: float = 0.0

    def should_reflect(self) -> bool:
        now = time.time()
        if now - self._last_reflect_time < self._reflection_interval:
            return False
        # 任意一个阶段有足够的数据就应该执行
        obs_count = self._store.count_memories(memory_type="observation")
        exp_count = self._store.count_memories(memory_type="experience")
        ins_count = self._store.count_memories(memory_type="insight")
        return (
            obs_count >= self._min_observations
            or exp_count >= self._min_experiences
            or ins_count >= self._min_insights
        )

    async def run_once(self) -> dict[str, Any]:
        """执行一轮反思。按优先级尝试三个阶段。"""
        # Stage 1: observation → experience
        observations = self._store.list_memories(
            memory_type="observation",
            limit=self._min_observations,
            offset=0,
        )
        if len(observations) >= self._min_observations and self._llm_complete:
            logger.info("Reflect Stage 1: %d observations → experiences", len(observations))
            try:
                result = await self._enrich_observations(observations)
                if result.get("enriched_count", 0) > 0:
                    self._last_reflect_time = time.time()
                    return result
            except Exception as e:
                logger.warning("Stage 1 failed: %s", str(e)[:500])
                import traceback
                logger.warning("Stage 1 traceback: %s", traceback.format_exc()[:500])

        # Stage 2: experience → insight
        experiences = self._store.list_memories(
            memory_type="experience",
            limit=self._min_experiences,
            offset=0,
        )
        if len(experiences) >= self._min_experiences:
            logger.info("Reflect Stage 2: %d experiences → insights", len(experiences))
            try:
                if self._llm_complete:
                    result = await self._extract_insights(experiences)
                else:
                    result = self._fallback_group_by_action(experiences, "insight")
                if result.get("insights") or result.get("models"):
                    self._last_reflect_time = time.time()
                    return result
            except Exception as e:
                logger.warning("Stage 2 failed: %s", e)

        # Stage 3: insight → mental_model
        insights = self._store.list_memories(
            memory_type="insight",
            limit=self._min_insights,
            offset=0,
        )
        if len(insights) >= self._min_insights and self._llm_complete:
            logger.info("Reflect Stage 3: %d insights → mental models", len(insights))
            try:
                result = await self._distill_mental_models(insights)
                if result.get("models"):
                    self._last_reflect_time = time.time()
                    return result
            except Exception as e:
                logger.warning("Stage 3 failed: %s", e)

        return {"models": [], "stage": None, "status": "skipped", "reason": "no stage ready"}

    # -- Stage 1: observation → experience ----------------------------------

    async def _enrich_observations(
        self, observations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """用 LLM 补全 observation 的结构化字段，转化为 experience。"""
        obs_compact = [
            {"index": i, "content": o.get("content", "")[:300]}
            for i, o in enumerate(observations)
        ]

        prompt = _PROMPT_OBSERVATION_ENRICH.format(
            observation_json=json.dumps(obs_compact, ensure_ascii=False, indent=2)
        )

        messages = [
            {"role": "system", "content": "你是一个经验分析师。始终返回严格 JSON。"},
            {"role": "user", "content": prompt},
        ]

        text = await self._llm_complete(messages)
        text = self._clean_json(text)

        try:
            enrichments = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Stage 1 LLM returned invalid JSON, raw=%s", text[:200])
            return {"enriched_count": 0, "status": "llm_json_error"}

        if not isinstance(enrichments, list):
            enrichments = [enrichments] if isinstance(enrichments, dict) else []

        enriched_count = 0
        for item in enrichments:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if isinstance(idx, str):
                try:
                    idx = int(idx)
                except (ValueError, TypeError):
                    idx = -1
            if idx is None or idx < 0 or idx >= len(observations):
                continue
            obs = observations[idx]
            try:
                action = item.get("action", "")
                context = json.dumps(item.get("context", {}), ensure_ascii=False)
                outcome = json.dumps(item.get("outcome", {}), ensure_ascii=False)
                confidence = min(max(item.get("confidence", 0.5), 0.0), 1.0)

                metadata = json.dumps({"enriched_confidence": confidence}, ensure_ascii=False)

                # 增量去重：检查已有 experience 是否语义相似
                existing_id = dedup_before_add(
                    self._store, self._embedding_client,
                    obs["content"], "experience",
                )
                if existing_id:
                    # 关联到已有的 experience 而非新建
                    self._store.add_edge(
                        source_id=obs["id"],
                        target_id=existing_id,
                        relation="enriched_to",
                    )
                    enriched_count += 1
                    logger.debug("  dedup: obs %d → existing experience %d", obs["id"], existing_id)
                    continue

                new_id = self._store.add_memory(
                    content=obs["content"],
                    embedding=None,
                    memory_type="experience",
                    mem_action=action or None,
                    mem_context=context,
                    mem_outcome=outcome,
                    mem_metadata=metadata,
                    parent_id=obs["id"],
                )
                if new_id:
                    self._store.add_edge(
                        source_id=obs["id"],
                        target_id=new_id,
                        relation="enriched_to",
                    )
                    enriched_count += 1
            except Exception as e:
                logger.warning("Skip enrich idx %d: %s", idx, e)

        return {
            "stage": 1,
            "enriched_count": enriched_count,
            "status": "ok",
        }

    # -- Stage 2: experience → insight --------------------------------------

    async def _extract_insights(
        self, experiences: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """用 LLM 聚类分析经验，提炼洞见。"""
        exps_compact = []
        for i, exp in enumerate(experiences):
            exps_compact.append({
                "index": i,
                "content": exp.get("content", "")[:200],
                "mem_action": exp.get("mem_action", ""),
                "mem_context": exp.get("mem_context", "{}"),
            })

        prompt = _PROMPT_EXPERIENCE_INSIGHT.format(
            experiences_json=json.dumps(exps_compact, ensure_ascii=False, indent=2)
        )

        messages = [
            {"role": "system", "content": "你是一个经验模式挖掘专家。始终返回严格 JSON。"},
            {"role": "user", "content": prompt},
        ]

        text = await self._llm_complete(messages)
        text = self._clean_json(text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Stage 2 LLM returned invalid JSON, using fallback")
            return self._fallback_group_by_action(experiences, "insight")

        raw_insights = data.get("insights", [])
        if not raw_insights:
            return self._fallback_group_by_action(experiences, "insight")

        models = []
        for ri in raw_insights:
            pattern = ri.get("pattern", "")
            confidence = min(max(ri.get("confidence", 0.5), 0.0), 1.0)
            indices = ri.get("supporting_indices", [])
            advice = ri.get("actionable_advice", "")

            if not pattern:
                continue

            full_content = pattern
            if advice:
                full_content += f"\n建议: {advice}"

            metadata = json.dumps({
                "confidence": confidence,
                "actionable_advice": advice,
                "source": "reflect_stage2",
            }, ensure_ascii=False)

            # 增量去重：检查是否已有相似 insight
            existing_id = dedup_before_add(
                self._store, self._embedding_client,
                full_content, "insight",
            )
            if existing_id:
                for idx in indices:
                    if 0 <= idx < len(experiences):
                        self._store.add_edge(
                            source_id=experiences[idx]["id"],
                            target_id=existing_id,
                            relation="supporting_evidence",
                        )
                models.append({
                    "model_id": existing_id,
                    "pattern": pattern,
                    "type": "insight",
                    "confidence": confidence,
                    "supporting_count": len(indices),
                })
                continue

            insight_id = self._store.add_memory(
                content=full_content,
                embedding=None,
                memory_type="insight",
                mem_metadata=metadata,
            )

            for idx in indices:
                if 0 <= idx < len(experiences):
                    self._store.add_edge(
                        source_id=experiences[idx]["id"],
                        target_id=insight_id,
                        relation="supporting_evidence",
                    )

            models.append({
                "model_id": insight_id,
                "pattern": pattern,
                "type": "insight",
                "confidence": confidence,
                "supporting_count": len(indices),
            })

        return {"stage": 2, "insights": models, "status": "ok"}

    # -- Stage 3: insight → mental_model ------------------------------------

    async def _distill_mental_models(
        self, insights: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """用 LLM 交叉分析洞见，蒸馏出心智模型。"""
        ins_compact = []
        for i, ins in enumerate(insights):
            ins_compact.append({
                "index": i,
                "pattern": ins.get("content", "")[:300],
                "metadata": ins.get("mem_metadata", "{}"),
            })

        prompt = _PROMPT_INSIGHT_MENTAL_MODEL.format(
            insights_json=json.dumps(ins_compact, ensure_ascii=False, indent=2)
        )

        messages = [
            {"role": "system", "content": "你是一个知识蒸馏专家。始终返回严格 JSON。"},
            {"role": "user", "content": prompt},
        ]

        text = await self._llm_complete(messages)
        text = self._clean_json(text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Stage 3 LLM returned invalid JSON")
            return {"models": [], "status": "llm_json_error"}

        raw_models = data.get("models", [])
        if not raw_models:
            return {"models": [], "status": "no_models_found"}

        models = []
        for rm in raw_models:
            name = rm.get("name", "")
            principle = rm.get("principle", "")
            applicability = rm.get("applicability", "")
            counter = rm.get("counter_indicators", "")
            confidence = min(max(rm.get("confidence", 0.5), 0.0), 1.0)
            indices = rm.get("supporting_indices", [])

            if not name or not principle:
                continue

            full_content = f"# {name}\n\n{principle}"
            if applicability:
                full_content += f"\n\n**适用场景**: {applicability}"
            if counter:
                full_content += f"\n\n**不适用场景**: {counter}"

            metadata = json.dumps({
                "model_name": name,
                "confidence": confidence,
                "applicability": applicability,
                "counter_indicators": counter,
                "source": "reflect_stage3",
            }, ensure_ascii=False)

            # 增量去重：检查是否已有相似 mental_model
            existing_id = dedup_before_add(
                self._store, self._embedding_client,
                full_content, "mental_model",
            )
            if existing_id:
                for idx in indices:
                    if 0 <= idx < len(insights):
                        self._store.add_edge(
                            source_id=insights[idx]["id"],
                            target_id=existing_id,
                            relation="supporting_evidence",
                        )
                models.append({
                    "model_id": existing_id,
                    "pattern": f"{name}: {principle[:100]}",
                    "type": "mental_model",
                    "confidence": confidence,
                    "supporting_count": len(indices),
                })
                continue

            model_id = self._store.add_memory(
                content=full_content,
                embedding=None,
                memory_type="mental_model",
                mem_metadata=metadata,
            )

            for idx in indices:
                if 0 <= idx < len(insights):
                    self._store.add_edge(
                        source_id=insights[idx]["id"],
                        target_id=model_id,
                        relation="supporting_evidence",
                    )

            models.append({
                "model_id": model_id,
                "pattern": f"{name}: {principle[:100]}",
                "type": "mental_model",
                "confidence": confidence,
                "supporting_count": len(indices),
            })

        return {"stage": 3, "models": models, "status": "ok"}

    # -- Fallback -----------------------------------------------------------

    def _fallback_group_by_action(
        self, items: list[dict[str, Any]], target_type: str = "insight"
    ) -> dict[str, Any]:
        """占位：按 mem_action 分组生成 insight 或 mental_model。"""
        clusters: dict[str, list[dict]] = {}
        for item in items:
            action = item.get("mem_action", "unknown")
            clusters.setdefault(action, []).append(item)

        models = []
        for action, group in clusters.items():
            if len(group) < 3:
                continue
            content = (
                f"模式识别：在「{action}」场景下，系统记录了 {len(group)} 条经验。"
                f"典型内容：{group[0].get('content', '')[:100]}"
            )
            new_id = self._store.add_memory(
                content=content,
                embedding=None,
                memory_type=target_type,
            )
            for item in group:
                self._store.add_edge(
                    source_id=item["id"],
                    target_id=new_id,
                    relation="supporting_evidence",
                )
            models.append({
                "model_id": new_id,
                "pattern": content,
                "type": target_type,
                "confidence": min(0.5 + len(group) * 0.02, 0.95),
                "supporting_count": len(group),
            })

        return {"stage": 2 if target_type == "insight" else 3, "insights": models, "status": "ok"}

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _clean_json(text: str) -> str:
        """清理 LLM 输出的 markdown JSON 包裹。"""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        return text.strip()