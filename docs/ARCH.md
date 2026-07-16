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
| POST | `/api/v1/search` | 检索，body 带 `namespace` |
| GET | `/api/v1/memories` | 分页，query 带 `namespace` |
| DELETE | `/api/v1/memories/:id` | 删除 |
| GET | `/api/v1/stats` | 统计，query 带 `namespace` |
| GET | `/api/v1/graph` | 图谱，query 带 `namespace` |
| POST | `/api/v1/reflect` | 触发反思 |
| GET | `/api/v1/mental-models` | 心智模型，query 带 `namespace` |

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