"""给现有文档型长文回填 doc_id 溯源标签。

判据（保守，只打有把握的）：
  - 【XXX】 开头的讲次/主题笔记
  - 《书名》开头的经典精读
两类之外的会话经验型长文【不打标】——它们不是外部文档，打标会误导检索。

打标效果（retriever 已实现，无需改代码）：
  - 检索不随时间衰减（time_weight = 1.0）
  - 不被 forget_advisor 建议归档
  - memory_type 保持不变，仍参与 reflect 提炼

用服务端连接（HybridMemoryStore）以保证与线上行为一致。
"""
from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, "/data/codes/hmem/server")

from engine.store import HybridMemoryStore  # noqa: E402

DRY_RUN = "--apply" not in sys.argv


def slug(s: str, n: int = 40) -> str:
    """生成稳定的 doc_id：保留中英文数字，其余转连字符。"""
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", s).strip("-")
    return s[:n]


def main() -> int:
    s = HybridMemoryStore(db_path="/data/hmem-data/default.db", embedding_dim=1024)
    s.initialize()
    c = s._conn
    c.execute("PRAGMA busy_timeout=60000")

    rows = c.execute(
        "SELECT id, content, memory_type, length(content) FROM memories "
        "WHERE length(content) >= 500 AND (doc_id = '' OR doc_id IS NULL) ORDER BY id"
    ).fetchall()

    plans = []
    for mid, content, mtype, _ln in rows:
        m = re.match(r"【([^】]{2,40})】", content) or re.match(r"《([^》]{2,50})》", content)
        if not m:
            continue  # 会话经验型，不打标
        title = m.group(1)
        plans.append((mid, title, slug(title), mtype))

    print(f"[+] 候选 {len(rows)} 条，其中文档型 {len(plans)} 条将被回填")
    by_doc: dict[str, list[int]] = {}
    for mid, title, did, _t in plans:
        by_doc.setdefault(did, []).append(mid)
    print(f"[+] 归入 {len(by_doc)} 个文档，多 chunk 文档:")
    for did, ids in sorted(by_doc.items(), key=lambda kv: -len(kv[1]))[:12]:
        if len(ids) > 1:
            print(f"    {did}: {len(ids)} 条 {ids}")

    if DRY_RUN:
        print("\n[预览模式] 加 --apply 执行")
        s.close()
        return 0

    updated = 0
    for mid, title, did, _t in plans:
        cur = c.execute(
            "UPDATE memories SET doc_id=?, doc_title=?, doc_category='文始道', "
            "updated_at=datetime('now','+8 hours') WHERE id=? AND (doc_id='' OR doc_id IS NULL)",
            (did, title, mid),
        )
        updated += cur.rowcount
    c.commit()
    print(f"[+] 已回填 {updated} 条")

    n = c.execute("SELECT COUNT(*) FROM memories WHERE doc_id != ''").fetchone()[0]
    print(f"[+] default 中带 doc_id 的条目: {n}")
    s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
