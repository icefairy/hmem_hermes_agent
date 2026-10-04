"""文本归一化 —— 借鉴 DeepSeek Engram 的「分词器压缩」(tokenizer compression)。

Engram 论文发现：子词分词器优先无损重构，导致 `Apple` / `apple` 被分到不相交的 ID，
N-gram 空间极度稀疏冗余；做一次 NFKC + 小写归一化后有效词表缩小 ~23%。

HMEM 的对应痛点：中文/英文/符号的同形异写（全角 vs 半角、大小写、圈码数字）
在 FTS5 里是**不同的 token**，导致「（」查不到「(」、「⑦」查不到「7」、「MCP」
查不到「mcp」。本模块提供一个统一的 `normalize()`，写入与查询两侧共用。

设计原则：
  - 只做**语义无损**的归一（不删词、不改词干），保证可回滚、不误导召回。
  - NFKC 统一兼容字符（全角→半角、圈码→数字、连字→分解）。
  - 折叠空白，避免多个空格/换行造成 token 差异。
  - 查询侧与写入侧必须调用同一函数，否则 FTS 仍然对不上。

用法：
    from engine.normalize import normalize
    normalize("ＭＣＰ（测试）")  →  "mcp(测试)"
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC 归一 + 小写 + 空白折叠。空串安全。"""
    if not text:
        return ""
    # NFKC: 全角→半角、圈码（⑦→7）、兼容连字（ﬁ→fi）等
    s = unicodedata.normalize("NFKC", text)
    s = s.lower()
    return _WS.sub(" ", s).strip()
