# Changelog

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.4.0] - 2026-10-04

### 新增

- **语义门控（Semantic Gate）** — 借鉴 DeepSeek Engram 的上下文感知门控（αₜ）思想：
  词面（FTS）命中但语义跑题的候选，按本地 HRR 相位相似度折扣其关键词分贡献，
  避免「含查询词但主题无关」的旧记忆刷高分。实测合成语料 MRR 0.833 → 1.000，
  真实库召回零损失。
  - 配置：`HMEM_GATE_ENABLED`（默认 true）、`HMEM_GATE_TAU`（0.1）、`HMEM_GATE_FLOOR`（0.5）
  - 详见 [`docs/SEMANTIC-GATE-EVAL.md`](docs/SEMANTIC-GATE-EVAL.md)
- **文本归一化（NFKC）** — FTS 分词前统一做 NFKC + 小写 + 空白折叠，消除全/半角、
  大小写、圈码（`⑦`↔`7`）不一致造成的漏检。收口于 `store._tokenize`（写入/查询共用）。
  - 存量回填端点：`POST /api/v1/backfill/tokenization?namespace=<ns>`
- **超长记忆分片** — 单条记忆超长时自动分片存储，检索时聚合回父记忆完整原文，
  修复 rerank 因超长文本整体 400 的问题；分片支持级联删除，避免孤儿。
- **doc_id 溯源字段** — 记忆携带来源文档标识，保护逻辑下沉到 store 层。

### 改进

- 记忆内容硬上限 8000 字符 + 去重合并总长上限，根治「雪球式」内容增长。
- 后台反思任务不再复用已关闭的 store 连接。
- 去重策略调整：不再硬删除，长文采用更严格阈值（修复课程笔记被误吞事故）。
- 补全 v5/v6 缺失的 schema 迁移（修复最近记忆列表空白）。

### 文档

- 新增 [`docs/ENGRAM-STUDY.md`](docs/ENGRAM-STUDY.md) — Engram → HMEM 适用性评估。
- 新增 [`docs/SEMANTIC-GATE-EVAL.md`](docs/SEMANTIC-GATE-EVAL.md) — 语义门控 A/B 实测记录。

## [0.3.0] - 2026-08-26

- 知识库支持：文档导入/CRUD、知识条目、库管理（piagent-hmem 插件新增 10 个工具）。
