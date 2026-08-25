# HMEM · 混合记忆系统架构

## 分库设计

每个 namespace 对应一个独立的 SQLite 文件，实现物理级隔离。

```
/data/hmem/
├── team-alpha.db       namespace: team-alpha     (代码审查/调试助手共享)
├── personal-duck.db    namespace: personal-duck  (个人助手私有)
├── agent-pilot.db      namespace: agent-pilot
└── ...
```

## 核心架构

```
 Hermes Agent ──HTTP──▶ HMEM Server (Docker)
                          │
                    FastAPI 路由
                          │
                 ┌────────┴────────┐
                 │                 │
           /api/v1/...      /api/v1/...
           带 ?namespace=    带 ?namespace=
                 │                 │
            ┌────┴────┐       ┌────┴────┐
            │team-alpha│      │personal-│
            │  .db     │      │ duck.db │
            └─────────┘       └─────────┘
            引擎:              引擎:
            store              store
            retriever          retriever
            embeddings         embeddings
```

## 存储层

每个 db 文件内：

```sql
memories       — 无 namespace 列，文件路径即为隔离
memories_fts   — FTS5 全文索引（jieba 分词）
vec_memories   — sqlite-vec 向量索引（1024 dim float32）
memory_edges   — 知识图谱边
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 探活 |
| POST | `/api/v1/memories` | 写入，body 带 `namespace` |
| POST | `/api/v1/search` | 检索，body 带 `namespace`（支持 `extra_namespaces` 多库合并） |
| GET | `/api/v1/memories` | 分页，query 带 `namespace` |
| DELETE | `/api/v1/memories/:id` | 删除 |
| GET | `/api/v1/stats` | 统计（含 `document_count`），query 带 `namespace` |
| GET | `/api/v1/graph` | 图谱，query 带 `namespace` |
| POST | `/api/v1/reflect` | 触发反思 |
| GET | `/api/v1/mental-models` | 心智模型，query 带 `namespace` |
| POST | `/api/v1/documents` | 导入知识库文档（自动分块 + 向量化 + 溯源） |
| GET | `/api/v1/documents` | 列出该库文档及 chunk 数 |
| GET | `/api/v1/documents/{doc_id}` | 取文档明细（含全部 chunk） |
| DELETE | `/api/v1/documents/{doc_id}` | 级联删除整篇文档 |

## 知识库角色（Memory + Knowledge 双角色）

HMEM 在记忆之外可同时充当**知识库**。核心是给 memories 表增加了文档语义（schema v4），
所有检索/向量/重排/图谱能力完全复用，无需独立知识库引擎：

```sql
memories.doc_id     — 文档 ID，chunk 归组键（空 = 普通记忆）
memories.doc_uri    — 文档来源 URI（可回引原文）
memories.doc_title  — 文档标题
memories.chunk_index — 分块序号（可还原顺序/定位）
memory_type = 'knowledge' — 知识类型，与 observation/experience/insight 分离
```

### 行为差异（知识 vs 记忆）

| 维度 | 普通记忆 | 知识条目（`knowledge` / 有 `doc_id`） |
|------|----------|------------------------------------------|
| 时间衰减 | 有（~35h 半衰期） | **无**（文档不看新旧） |
| 反思蒸馏 | 会被 LLM 聚类成心智模型 | **不参与 reflect**（独立类型，天然隔离） |
| 溯源 | 无 | `source: {doc_id, title, uri, chunk_index}` 透出 |
| 管理 | 单条 | 文档级 CRUD（导入/列表/级联删除） |

时间衰减豁免在 `retriever._compute_score` 实现：`memory_type == "knowledge"` 或带 `doc_id` 时 `time_weight = 1.0`。
reflect 各阶段按 memory_type 精确筛选，`knowledge` 不会被捞进任何 stage。

### 多库检索与权重

`POST /api/v1/search` 的 `extra_namespaces` 支持两种形态（向后兼容）：

```bash
# 纯字符串（weight = 1.0）
curl -X POST .../api/v1/search -d '{"namespace":"app", "extra_namespaces":["kb-eng"]}'

# 对象形态（可指定权重，如共享记忆 0.8 / 知识库 1.0）
curl -X POST .../api/v1/search -d '{"namespace":"app", "extra_namespaces":[{"ns":"kb-eng","weight":1.0}]}'
```

命中结果带 `_ns`（来源库）与 `extra_weight`/`shared` 标注，方便上层区分记忆与知识。

### 文档导入示例

```bash
curl -X POST .../api/v1/documents -H 'Content-Type: application/json' -d '{
  "content": "部署手册全文……",
  "title": "K8s 部署手册",
  "uri": "https://docs.example.com/k8s",
  "doc_id": "K8S-DEPLOY",      # 可选；不带自动生成
  "namespace": "kb-eng",
  "chunk_size": 800, "overlap": 80
}'
```

导入时自动：分块 → 批量向量化 → 逐块写入（共享 doc_id）→ **关闭该库自动 reflect**（保真）。
知识库的命名约定建议用 `kb-*` 前缀（如 `kb-eng-docs`），与业务记忆 namespace 区分。

## Hermes 插件

```yaml
plugins:
  hmem:
    api_url: http://hmem:8000
    api_key: sk-xxx
    namespace: team-alpha    ← 必填，路由到对应 db
```

多个 agent 配置相同 namespace → 共享记忆。

## 与 Hindsight 对比

| 维度 | Hindsight | HMEM |
|------|-----------|------|
| 存储 | PostgreSQL + 向量库 | **SQLite + sqlite-vec** |
| 隔离 | schema/user | **分库（文件级）** |
| 部署依赖 | PG 集群 | **零依赖，docker volume 即可** |
| 插件集成 | HindsightWrapper | **Hermes MemoryProvider** |
| 反思引擎 | Rust 高性能 | Python（可配置间隔） |
| 可视化 | 无原生 WebUI | **内置 Vue3 SPA** |
| 知识图谱 | 有 | **有（ECharts 力导向图）** |