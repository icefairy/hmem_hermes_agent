"""向量化去重与反思线程隔离的回归测试 —— 2026-11-09 写入 208 秒事故。

事故现场：一次 hmem_write 耗时 208 秒，而同期 embed 本身只花 0.5 秒。
两层根因：
  1. merge_similar 的贪心聚类内层用纯 Python 逐维算余弦，
     O(n_items × n_clusters × dim)。piagent 库 1600 条 / 500 簇 / 1024 维
     实测要 **108 秒**。改成 numpy 矩阵乘后约 1 秒。
  2. 反思（去重 + LLM 归纳）直接在 uvicorn 的事件循环线程上跑，
     期间所有写入请求排队。

本文件锁定三条不变量：
  1. 向量化路径与纯 Python 路径的聚类判定完全一致（改造不能改变去重结果）
  2. numpy 不可用时能安全降级到纯 Python
  3. 反思进程级互斥：并发进来时后来的直接跳过（不排队、不互抢 DB）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import pytest

import engine.dedup as dedup
from engine.reflect import reflect_in_thread


DIM = 64


def _rand_vec(rng: random.Random) -> list[float]:
    return [rng.random() for _ in range(DIM)]


def _make_clusters(rng: random.Random, n: int) -> list[dict]:
    return [
        {"centroid": _rand_vec(rng), "ids": [i], "contents": [f"c{i}"]}
        for i in range(n)
    ]


def test_vectorized_matches_pure_python():
    """向量化路径必须与纯 Python 逐簇计算结果一致（含并列最高分的取舍）。"""
    np = dedup._numpy()
    if np is None:
        pytest.skip("环境无 numpy，向量化路径未启用")

    rng = random.Random(20261109)
    clusters = _make_clusters(rng, 40)

    # 复刻 merge_similar 里的矩阵构建方式
    mat = np.array([c["centroid"] for c in clusters], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    for _ in range(100):
        emb = _rand_vec(rng)
        py_cluster, py_sim = dedup._best_cluster_loop(emb, clusters)

        v = np.asarray(emb, dtype=np.float32)
        vn = float(np.linalg.norm(v))
        sims = mat @ (v / vn)
        bi = int(np.argmax(sims))
        np_sim = float(sims[bi])
        np_cluster = clusters[bi] if np_sim > 0 else None

        assert (py_cluster is None) == (np_cluster is None)
        if py_cluster is not None:
            assert py_cluster["contents"] == np_cluster["contents"]
        assert abs(py_sim - np_sim) < 1e-4


def test_degrades_without_numpy(monkeypatch):
    """numpy 不可用时 _numpy() 返回 None，聚类判定走纯 Python 分支不报错。"""
    monkeypatch.setattr(dedup, "_NP_CACHE", dedup._NP_UNSET)
    # 模拟 import numpy 失败
    monkeypatch.setitem(sys.modules, "numpy", None)
    assert dedup._numpy() is None

    rng = random.Random(7)
    clusters = _make_clusters(rng, 5)
    best, sim = dedup._best_cluster_loop(_rand_vec(rng), clusters)
    assert best is not None
    assert 0.0 <= sim <= 1.0


def test_reflect_mutex_skips_second_caller():
    """反思进程级互斥：锁被占时后来的调用直接跳过，不排队。"""

    class _FakeEngine:
        def run_once(self):  # pragma: no cover — 不该被调用
            raise AssertionError("锁被占时不应执行 run_once")

    # 手动占住锁，模拟已有一轮在跑
    from engine import reflect as reflect_mod

    assert reflect_mod._REFLECT_LOCK.acquire(blocking=False)
    try:
        result = reflect_in_thread(_FakeEngine())
        assert result.get("skipped") is True
        assert result.get("reason") == "already_running"
    finally:
        reflect_mod._REFLECT_LOCK.release()


def test_reflect_mutex_releases_after_error():
    """run_once 抛异常时锁必须释放，否则后续反思永久失效。"""
    from engine import reflect as reflect_mod

    class _BoomEngine:
        def run_once(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        reflect_in_thread(_BoomEngine())

    # 锁应已释放：能再拿到
    assert reflect_mod._REFLECT_LOCK.acquire(blocking=False)
    reflect_mod._REFLECT_LOCK.release()
