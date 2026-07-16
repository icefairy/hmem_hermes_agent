# HMEM · 混合记忆系统升级规划

> 对标 Hindsight 仿生记忆架构，让 Hermes Agent 从"记忆存储"进化到"认知学习"。

---

## 一、当前架构（v0.1 · 已交付）

```
 Hermes Agent
      │
      ▼
 HybridMemoryProvider (MemoryProvider ABC)
      │
      ├── store.py         SQLite + FTS5 + sqlite-vec
      ├── retriever.py     关键词 + 向量 → 重排
      └── embeddings.py    bge-m3 / rerankv2m3 客户端
```

**能力**：FTS5 全文检索 + sqlite-vec 向量相似性 + rerank 重排
**缺**：无时间衰减、无图关系、无 Reflect 循环、无心智模型层

---

## 二、目标架构（Hindsight 仿生设计）

```
 ┌─────────────────────────────────────────────────────────┐
 │                     HMEM 引擎                            │
 │                                                          │
 │  三层记忆模型                                             │
 │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
 │  │ 世界事实层 │  │  经验层   │  │心智模型层 │              │
 │  │ (静态知识) │←─│(交互记录)│←─│(抽象模式)│              │
 │  └──────────┘  └──────────┘  └──────────┘              │
 │       ↑             ↑             ↑                     │
 │       └────── 四维并行检索 ──────┘                     │
 │        语义 · 关键词 · 图关系 · 时间范围                │
 │                    │                                    │
 │              交叉编码器重排序                             │
 │                    │                                    │
 │         Reflect 引擎（定期反思 → 抽象心智模型）           │
 │                                                          │
 │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
 │  │ Hermes   │  │  WebUI   │  │ REST API │              │
 │  │ 插件入口  │  │ Vue3 SPA │  │ FastAPI  │              │
 │  └──────────┘  └──────────┘  └──────────┘              │
 │       ↑             ↑             ↑                     │
 │       └───── 共用 Sqlite 数据库 ─────┘                 │
 └─────────────────────────────────────────────────────────┘
```

---

## 三、升级路线

### Phase 1 — 结构化元数据 & 时间衰减（1天）

**现状**：记忆存储为纯文本 `(id, content, content_jieba, agent_space, created_at)`

**升级内容**：

```sql
-- 新 schema（向后兼容）
CREATE TABLE memories_v2 (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    content      TEXT NOT NULL,          -- 原始内容（不变）
    content_jieba TEXT,                   -- jieba 分词（不变）
    mem_action   TEXT DEFAULT '',         -- "code_generation", "qa", "debug", ...
    mem_context  TEXT DEFAULT '{}',       -- JSON: {language, framework, ...}
    mem_outcome  TEXT DEFAULT '{}',       -- JSON: {success, feedback, ...}
    mem_metadata TEXT DEFAULT '{}',       -- JSON: {agent_version, session_id, ...}
    agent_space  TEXT NOT NULL DEFAULT 'default',
    memory_type  TEXT NOT NULL DEFAULT 'experience',  -- 'fact' | 'experience' | 'mental_model'
    parent_id    INTEGER DEFAULT NULL,    -- 关联上级经验（心智模型溯源）
    hit_count    INTEGER DEFAULT 0,       -- 被检索次数（热数据标志）
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
);
```

**时间衰减权重**：检索时对 `created_at` 应用指数衰减

```python
time_weight = exp(-λ · (now - created_at))
# λ 可配置，默认 0.01（≈ 100 天半衰期）
```

**输出**：
- [ ] store.py schema 升级（v1→v2 迁移脚本）
- [ ] retriever.py 增加时间衰减因子
- [ ] Hermes tool 适配新字段

---

### Phase 2 — 四维并行检索（2天）

**现状**：二维检索（FTS5 关键词 + vec 语义）

**升级**：

#### 维度 3：图关系检索

```sql
-- 记忆关联表
CREATE TABLE memory_edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id    INTEGER NOT NULL REFERENCES memories(id),
    target_id    INTEGER NOT NULL REFERENCES memories(id),
    relation     TEXT NOT NULL,  -- 'similar', 'causal', 'contains', 'contradicts'
    weight       REAL DEFAULT 1.0,
    created_at   TEXT NOT NULL
);
```

图检索策略：

1. **直接邻居**：`source_id = X OR target_id = X`（0 跳）
2. **二度关联**：通过中间记忆间接连接（1 跳）
3. **关系语义**：特定关系的路径，如 `causal` 链溯源根因

#### 维度 4：时间范围检索

- 滑动窗口：`last_N_days` 最近 N 天
- 周期性识别：分析同一时段的记忆密度（用户活跃周期）
- 热数据缓存：`hit_count > threshold` 的记忆驻留内存

#### 并行调度

```
          ┌── FTS5 关键词     ──┐
 Query ──→├── vec 语义搜索    ──┤──→ 合并/去重 ──→ rerank ──→ Top-K
          ├── 图关系遍历      ──┤
          └── 时间衰减+窗口   ──┘
```

**输出**：
- [ ] `store.py` 增加 `memory_edges` 表 + 关联检索方法
- [ ] `retriever.py` 增加四维并行调度
- [ ] 图关系自动构建（相似记忆自动加边）

---

### Phase 3 — Reflect 引擎（3-5天，核心升级）

**这是"记忆"与"学习"的本质区别。**

#### 触发机制

```python
class ReflectEngine:
    def __init__(self):
        self.min_experiences = 50       # 最少经验条数才触发
        self.reflection_interval = 3600 # 秒，默认 1 小时
        self._last_reflect_time = 0
    
    def should_reflect(self, store) -> bool:
        """判断是否需要执行一轮反思"""
        now = time.time()
        if now - self._last_reflect_time < self.reflection_interval:
            return False
        total = store.count_memories(memory_type='experience')
        return total >= self.min_experiences
    
    def reflect(self, store, llm_client) -> list[dict]:
        """执行反思：聚类 → 抽象 → 写入心智模型层"""
        pass
```

#### 反思流程

```
1. 批量读取最新的 / 高分的 N 条经验
2. 用 LLM 聚类分析（或本地聚类）：
     "以下经验中，有哪些共同模式？"
3. 对每个聚类，生成心智模型：
     input:  [经验1, 经验2, 经验3, ...]
     output: {
       "pattern": "用户在碰到XX类问题时偏好XX方案",
       "confidence": 0.85,
       "supporting_evidence": [id1, id3, id7],
       "actionable_advice": "下次遇到同类问题时，优先尝试XX"
     }
4. 写入心智模型层（memory_type = 'mental_model'）
5. 更新记忆关联（parent_id = 心智模型 ID）
```

#### 触发方式

- **时机 1**：Hermes `on_session_end` 时检查条件
- **时机 2**：独立的 cronjob `hmem-reflect`，每小时检查
- **时机 3**：手动触发：`hybrid_memory_reflect`

**输出**：
- [ ] `reflect.py` 反思引擎
- [ ] `store.py` 增加 `memory_type` 过滤 + 批量读取
- [ ] LLM 聚类 prompt 模板
- [ ] Cronjob 注册
- [ ] Hermes `hybrid_memory_reflect` tool

---

### Phase 4 — WebUI 可视化（并行，2天）

**目标**：用户可通过浏览器查看/搜索/管理记忆，可视化三层结构和反思结果。

#### 页面设计

| 路由 | 功能 |
|------|------|
| `/` | 概览：统计数据 + 最近记忆列表 |
| `/search` | 混合检索，展示各维度贡献度 |
| `/graph` | 记忆关系图（force-directed graph） |
| `/reflect` | 心智模型列表、反思历史 |
| `/memories/:id` | 单条记忆详情（含关联的心智模型） |

#### 技术栈

```
Vue 3 + Element Plus + ECharts + FastAPI（后端代理）
```

FastAPI 后端提供：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/stats` | GET | 统计数据 |
| `/api/search` | POST | 混合检索（支持指定维度权重） |
| `/api/write` | POST | 写入记忆 |
| `/api/delete/:id` | DELETE | 删除 |
| `/api/list` | GET | 分页列表 |
| `/api/graph` | GET | 记忆关系图数据 |
| `/api/reflect` | POST | 手动触发反思 |
| `/api/mental-models` | GET | 心智模型列表 |
| `/api/health` | GET | 探活 |

**输出**：
- [ ] FastAPI 后端（`webui/server.py`）
- [ ] Vue3 前端（`webui/src/`）
- [ ] 关系图可视化（ECharts / D3 force graph）
- [ ] Docker Compose 一键启动

---

## 四、项目目录结构（完成态）

```
/root/codes/hmem/
├── hermes-agent/               # → Hermes Agent 插件部分
│   └── plugins/
│       └── hybrid-memory/
│           ├── __init__.py     # MemoryProvider 入口
│           ├── store.py        # SQLite 存储层
│           ├── retriever.py    # 四维并行检索
│           ├── embeddings.py   # bge-m3 / rerank 客户端
│           └── reflect.py      # 反思引擎 (Phase 3)
├── webui/                      # → 可视化部分
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── main.ts
│   │   ├── App.vue
│   │   ├── views/
│   │   │   ├── Dashboard.vue
│   │   │   ├── Search.vue
│   │   │   ├── Graph.vue
│   │   │   └── Reflect.vue
│   │   └── components/
│   └── server.py               # FastAPI 后端代理
├── docs/
│   ├── PLAN.md                 # 本文件
│   └── ADR-*.md                # 架构决策记录
├── .gitignore
└── README.md
```

---

## 五、优先级建议

```
Phase 1 (结构化 + 时间)  ──→  Phase 3 (Reflect)  ──→  Phase 4 (WebUI)
         │                               │
         └── Phase 2 (四维检索) ←─────────┘
```

**起步建议**：Phase 1 → Phase 3 的核心循环 → Phase 4 可视化 → Phase 2 锦上添花。

因为 Reflect 是 Hindsight 的本质差异化能力，需要尽早验证；而四维检索在已有二维检索基础上增量改进不大，可放在后面。

---

## 六、升级风险与注意事项

| 风险 | 缓解措施 |
|------|---------|
| 旧数据库无法回退 | Phase 1 用 v1→v2 迁移脚本，保留快照 |
| Reflect 调用 LLM 成本 | 可配置 `reflection_interval` 和 `min_experiences`，或用本地模型替代 |
| 图关系存储膨胀 | 仅保留 `similar` 关系的自动边，`causal`/`contradicts` 由 Reflect 引擎按需生成 |
| WebUI 暴露敏感记忆 | 默认 localhost-only，后续支持 token 鉴权 |