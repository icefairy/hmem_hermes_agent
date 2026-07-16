# HMEM · 升级规划 v2

> 架构变更：同表 namespace → 分库方案。Phase 1-4 升级路线。

---

## 一、当前架构（v0.2 — 分库方案）

```spec/core/overview.md
每个 namespace 一个独立的 SQLite 文件

/data/hmem/
├── team-alpha.db       namespace: team-alpha
├── personal-duck.db    namespace: personal-duck
└── ...

表结构（每个 db 文件内）：
memories       — 核心记忆表（无 namespace 列）
memories_fts   — FTS5 全文索引
vec_memories   — sqlite-vec 向量索引
memory_edges   — 图关系边表
```

#### 核心变化

- `namespace` 不再存为列 → 通过文件路径隐式隔离
- Hermes 插件只需配置 `namespace`，API 路由到对应 db
- 共享记忆 = 多个 agent 指向同一 namespace
- 所有 SQL 去掉 `WHERE namespace = ?`，更简单

---

## 二、升级路线（对标 Hindsight）

### Phase 1 — 结构化元数据 & 时间衰减（1 天）

**现状**：记忆为纯文本 `(id, content, content_jieba, created_at)`

**升级**：

```sql
-- v3 schema（向后兼容 v2）
ALTER TABLE memories ADD COLUMN memory_type   TEXT DEFAULT 'experience';
-- 'fact' | 'experience' | 'mental_model'
ALTER TABLE memories ADD COLUMN mem_action    TEXT DEFAULT '';
ALTER TABLE memories ADD COLUMN mem_context   TEXT DEFAULT '{}';
ALTER TABLE memories ADD COLUMN mem_outcome   TEXT DEFAULT '{}';
ALTER TABLE memories ADD COLUMN mem_metadata  TEXT DEFAULT '{}';
ALTER TABLE memories ADD COLUMN parent_id     INTEGER DEFAULT NULL;
ALTER TABLE memories ADD COLUMN hit_count     INTEGER DEFAULT 0;
ALTER TABLE memories ADD COLUMN updated_at    TEXT;
```

**时间衰减权重**：

```python
time_weight = exp(-λ · hours_ago)
# λ 可配置，默认 0.02（≈ 35 小时半衰期）
```

#### Del

- [ ] store.py schema 升级（idempotent ALTER TABLE）
- [ ] retriever.py 增加时间衰减因子
- [ ] Hermes tool 适配新字段（hmem_write 支持 mem_action/context/outcome）

---

### Phase 2 — 四维并行检索（2 天）

**现状**：二维检索（FTS5 关键词 + vec 语义）

**新增维度**：

#### 维度 3：图关系检索

```sql
-- 已存在 memory_edges 表
SELECT target_id FROM memory_edges
WHERE source_id IN (子查询) AND relation = 'similar'
```

#### 维度 4：时间范围检索

- 滑动窗口：`WHERE created_at > datetime('now', '-N days')`
- 热数据缓存：`hit_count > threshold` 的记忆驻留内存

```
Query → FTS5 关键词
      → vec 语义搜索
      → 图关系遍历
      → 时间衰减+窗口
      → 合并/去重 → rerank → Top-K
```

#### Del

- [ ] `retriever.py` 四维并行调度
- [ ] 检索结果增加维度来源标注
- [ ] 可配置维度权重

---

### Phase 3 — Reflect 引擎（3-5 天，核心升级）

**"记忆"与"学习"的本质区别。**

#### 流程

```python
class ReflectEngine：
    1. 批量读取最新的 / 高分的 N 条经验
    2. 用 LLM 聚类分析：
       "以下经验中，有哪些共同模式？"
    3. 对每个聚类，生成心智模型：
       pattern: str          # "用户偏好XX方案"
       confidence: float     # 0-1
       supporting_ids: list  # 关联经验 ID
       actionable_advice: str
    4. 写入心智模型层（memory_type = 'mental_model'）
    5. 创建 memory_edges（relation = 'supporting_evidence'）
```

#### 触发方式

- 时机 1：Hermes `on_session_end`
- 时机 2：cronjob `hmem-reflect`，每小时检查
- 时机 3：手动 `POST /api/v1/reflect`

#### Del

- [ ] `reflect.py` 引擎（已完成框架，待注入 LLM 回调）
- [ ] Hermes `hmem_reflect` tool

---

### Phase 4 — WebUI 可视化（并行，2 天）

**目标**：浏览器查看/搜索/管理记忆，可视化和反思结果。

#### 路由

| 路由 | 功能 |
|------|------|
| `/` | 概览：统计 + 最近记忆列表 |
| `/search` | 混合检索，展示各维度贡献度 |
| `/graph` | 记忆关系图（force-directed graph） |
| `/reflect` | 心智模型列表、反思历史 |
| `/memories/:id` | 单条记忆详情 |

#### 后端 API

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/v1/stats?namespace=x` | 统计数据 |
| POST | `/api/v1/search` | 混合检索 |
| POST | `/api/v1/memories` | 写入 |
| GET | `/api/v1/memories?namespace=x` | 分页列表 |
| DELETE | `/api/v1/memories/:id` | 删除 |
| GET | `/api/v1/graph?namespace=x` | 关系图数据 |
| POST | `/api/v1/reflect` | 触发反思 |
| GET | `/api/v1/mental-models?namespace=x` | 心智模型列表 |

#### Del

- [ ] WebUI 搜索页面（Vue3 + Element Plus）
- [ ] 知识图谱可视化（ECharts force graph）
- [ ] FastAPI 后端 + 静态文件挂载

---

## 三、项目结构

```
hmem/
├── server/                    Docker 部署
│   ├── main.py               FastAPI 入口
│   ├── config.py             环境变量配置
│   ├── middleware.py          Bearer token
│   ├── engine/               记忆引擎核心
│   │   ├── store.py          SQLite + FTS5 + vec0（分库）
│   │   ├── retriever.py      四维并行检索
│   │   ├── embeddings.py     bge-m3 / rerank 客户端
│   │   └── reflect.py        反思引擎
│   ├── routers/              REST API
│   ├── webui/               Vue3 SPA
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── hermes-plugin/            Hermes 轻量插件
│   └── __init__.py           仅 HTTP 客户端，~300 行
│
└── docs/
    ├── PLAN.md                  本文件
    └── ARCH.md                  架构设计
```

---

## 四、部署

```yaml
services:
  hmem:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - hmem-data:/data/hmem
    environment:
      HMEM_API_KEY: ${HMEM_API_KEY}
      EMBEDDING_BASE_URL: ${EMBEDDING_BASE_URL}
      EMBEDDING_API_KEY: ${EMBEDDING_API_KEY}
      EMBEDDING_MODEL: bge-m3
      RERANK_MODEL: rerankv2m3
      REFLECT_INTERVAL: 3600
      REFLECT_MIN_EXPERIENCES: 50

volumes:
  hmem-data:
```