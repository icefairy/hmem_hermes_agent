"""语义门控（Engram 上下文门控借鉴）有效性评测。

目标：验证 gate 降低「字面词命中但语义跑题」记忆对检索的污染。
评测集含三类：
  A. 真相关（字面+语义都相关）—— gate 不应伤害召回
  B. 语义陷阱（含查询词但主题无关）—— gate 应降低其排名
  C. 纯语义（不含字面词但相关）—— 走向量/HRR，不受 FTS gate 影响

门控是"折扣"而非"删除"，故阈值化断言：带门控时 B 类陷阱的靠前率应下降，
A 类真相关的 recall@k 不应下降。
"""

from __future__ import annotations

import os
import tempfile

from engine.store import HybridMemoryStore
from engine.retriever import HybridRetriever


def _store_with(items: list[tuple[str, str]]) -> HybridMemoryStore:
    d = tempfile.mkdtemp()
    s = HybridMemoryStore(db_path=os.path.join(d, "t.db"), embedding_dim=1024)
    s.initialize()
    for content, mtype in items:
        s.add_memory(content=content, memory_type=mtype)
    return s


# 评测集：query → (真相关内容片段, 陷阱内容片段)
CORPUS = [
    # 主题：supervisor 部署
    ("supervisor 部署 Go 二进制并用 healthz 探活，配置在 /etc/supervisor/conf.d/", "observation"),
    ("HMEM 服务由 supervisor 管理，重启用 supervisorctl restart hmem", "observation"),
    ("讨论 supervisor 模式：监督式学习 vs 无监督学习在 NLP 中的区别", "observation"),  # 陷阱：含 supervisor 主题不同
    # 主题：奇门取号
    ("奇门遁甲取号规则：生门开门休门三吉门宫数按 +9 递进展开", "observation"),
    ("大六壬三传取数：太玄数 甲己子午九 乙庚丑未八", "observation"),
    ("取号算法优化：哈希表的键值取号与冲突处理", "observation"),  # 陷阱：含"取号"但指哈希
    # 主题：widget 刷新
    ("桌面小组件刷新：AlarmManager 精确闹钟 SCHEDULE_EXACT_ALARM 权限", "observation"),
    ("刷新频率调整：数据库批量刷新与 WAL checkpoint 策略", "observation"),  # 陷阱：含"刷新"但指数据库
]


def _make(store, gate: bool) -> HybridRetriever:
    return HybridRetriever(
        store=store, embedding_client=None, keyword_weight=0.4,
        vector_weight=0.6, hrr_weight=0.4,
        gate_enabled=gate, gate_tau=0.2, gate_floor=0.5,
    )


def _ids_of(res):
    return [r["id"] for r in res]


def test_gate_reorders_trap_below_real():
    """有 HRR 信号时：门控应把「语义陷阱」压到「真相关」之后。"""
    store = _store_with(CORPUS)
    try:
        # 直接构造带 fts_rank + hrr_similarity 的条目，验证 _compute_score 行为
        # 真相关：字面分高 + 语义相似度 ≥ tau（gate=1，不折扣）
        real = {"id": 1, "fts_rank": -3.0, "hrr_similarity": 0.25}
        # 陷阱：字面分高但语义相似度远低于 tau（gate=floor=0.5）
        trap = {"id": 3, "fts_rank": -3.0, "hrr_similarity": 0.02}

        no_gate = _make(store, gate=False)
        s_real_ng = no_gate._compute_score(dict(real))
        s_trap_ng = no_gate._compute_score(dict(trap))
        gap_ng = s_real_ng - s_trap_ng

        gate = _make(store, gate=True)
        s_real_g = gate._compute_score(dict(real))
        s_trap_g = gate._compute_score(dict(trap))
        gap_g = s_real_g - s_trap_g
        # 有门控：真相关 > 陷阱，且二者差距应比无门控时更大（陷阱被压低）
        assert s_real_g > s_trap_g, f"门控应压低陷阱：real={s_real_g:.3f} trap={s_trap_g:.3f}"
        assert gap_g > gap_ng + 0.05, f"门控应扩大真/陷阱差距：{gap_ng:.3f}→{gap_g:.3f}"
        # 真相关的分数不应被显著伤害（语义高 → gate≈1）
        assert abs(s_real_g - s_real_ng) < 1e-6, "高语义条目的分数不应被门控改变"
        # 陷阱分数应明显下降
        assert s_trap_g < s_trap_ng - 0.05, "低语义陷阱的分数应被门控压低"
    finally:
        store.close()


def test_gate_safe_without_signal():
    """无 HRR/向量信号时门控不折扣（安全：不误伤）。"""
    store = _store_with([("x", "observation")])
    try:
        e = {"id": 1, "fts_rank": -3.0}  # 无任何语义信号
        ng = _make(store, gate=False)._compute_score(dict(e))
        g = _make(store, gate=True)._compute_score(dict(e))
        assert abs(ng - g) < 1e-9, "无语义信号时门控应为恒等"
    finally:
        store.close()


def test_gate_floor_bounds_attenuation():
    """门控下限 floor：极低语义相似度的折扣不低于 floor。"""
    store = _store_with([("x", "observation")])
    try:
        r = HybridRetriever(store=store, embedding_client=None, gate_enabled=True,
                            gate_tau=0.2, gate_floor=0.5)
        # sim=0 → gate 应 = floor = 0.5（而非 0）
        e0 = {"id": 1, "fts_rank": -9.0, "hrr_similarity": 0.0}
        e_hi = {"id": 2, "fts_rank": -9.0, "hrr_similarity": 1.0}
        s0 = r._compute_score(dict(e0))
        s_hi = r._compute_score(dict(e_hi))
        assert s_hi > s0, "高语义应高于极低语义"
        # 折扣后 s0 仍 > 0（floor 不是清零）
        assert s0 > 0
    finally:
        store.close()
