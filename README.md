# HMEM · Hybrid Memory Engine / 混合记忆系统

> **让 AI Agent 拥有长期记忆和持续学习能力**  
> **Give AI Agents long-term memory and the ability to learn from experience**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com)

**English** | [中文](#中文版)

---

## English

HMEM (Hybrid Memory) is a **biomimetic memory engine** for AI Agents. Inspired by [Hindsight](https://github.com/vectorize-io/hindsight), it goes beyond simple fact recall — it **learns patterns from experience and forms transferable mental models**.

SQLite + sqlite-vec + jieba + bge-m3. Zero external dependencies. One Docker container.

### ✨ Key Features

#### 🧠 True Learning, Not Just Retrieval

Most memory systems are just "information retrievers" — store what you give, return what you query. HMEM is different.

| Capability | HMEM | Traditional RAG | Vector DB |
|-----------|------|-----------------|-----------|
| Full-text keyword search | ✅ FTS5 + jieba (Chinese) | ✅ | ❌ |
| Semantic vector search | ✅ bge-m3 1024-dim | ✅ | ✅ |
| Cross-encoder re-ranking | ✅ rerankv2m3 | ❌ | ❌ |
| **Time-decay weighting** | ✅ Recent memories weighted higher | ❌ | ❌ |
| **Knowledge graph relations** | ✅ memory_edges network | ❌ | ❌ |
| **Reflection → Mental models** | ✅ LLM clustering → abstract patterns | ❌ | ❌ |
| **Visual management** | ✅ Vue3 WebUI (graph/search/reflect) | ❌ | ❌ |

#### 🔥 Reflect Engine: From Experience to Wisdom

This is what sets HMEM apart from every other memory solution.

```
Write memories                  Reflect loop                    Retrieval
━━━━━━━━━━━━━━━━            ━━━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━━━
User: "Wrote a file         ┌─ Accumulate 50+ ─┐          Not just facts,
      upload API with       │   experiences    │          but patterns:
      FastAPI"              │                   │          "This user prefers
User: "PostgreSQL           │  LLM clustering  │          detailed technical
      query is slow…"       │   & abstraction  │          explanations"
User: "User feedback        │  └─ Mental model ─┘          "
      says comments         →   "This user
      aren't detailed"      │    prefers detailed
User: "Deploying to K8s     │    technical docs
      has networking …"     │    with code samples"        Response adapts style
                            │                   │
                      Mental model guides future behavior
```

**The result**: The more you use it, the better it understands you — not through prompt engineering, but genuine learning from interaction.

#### 🚀 Hybrid Retrieval: Four Dimensions in Parallel

```
Keyword search (FTS5)  ────→ Exact match for "pgvector"
Vector semantic (bge-m3) ──→ "DB slow" → finds "PostgreSQL query optimization"
Graph relationships    ────→ "Last time's issue" → linked debug records
Time decay             ────→ Recent memories weighted higher
         │
         ▼
  Cross-encoder re-rank  ──→ Not simple score averaging, but semantic reordering
         │
         ▼
  Final Top-K results
```

#### 🏗️ Physical Isolation, Multi-Agent by Design

Each **namespace** is an independent SQLite database file.

```
/data/hmem/
├── team-alpha.db       ← code-review / debug-assistant shared memory
├── personal-duck.db    ← Personal assistant private
└── agent-pilot.db      ← Another independent agent
```

Same namespace → shared memory and mental models. Different namespaces → complete physical isolation. No complex auth, no PostgreSQL schema — file-level isolation, clean and simple.

#### 📦 Zero External Dependencies

| Dependency | HMEM | PostgreSQL-based |
|-----------|------|-----------------|
| Database | **SQLite** (single file) | PostgreSQL cluster |
| Vector engine | **sqlite-vec** (embedded) | pgvector extension |
| Middleware | **None** | Redis / message queue |
| Deployment | **docker compose up** | 3+ containers |
| Backup | **cp one file** | pg_dump full |

**One Docker container, ready out of the box.** No heavy ops burden.

#### 🖥️ Visual Brain: WebUI Dashboard

Built-in Vue3 + ECharts management interface.

- **📊 Overview**: Memory count, type distribution, embedding status at a glance
- **🔍 Search**: Hybrid retrieval with source annotation (keyword/vector/rerank)
- **🕸️ Knowledge Graph**: Force-directed graph of memory relationships; mental models highlighted
- **🧠 Reflection**: Manual trigger, mental model list with supporting experiences
- **⚙️ Settings**: Configurable thresholds, night-time auto-reflection window

### Architecture

```
hmem/
├── hermes-plugin/         Hermes Agent plugin (~300 lines HTTP API client)
│   └── __init__.py        Just configure api_url + api_key + namespace
├── server/                HMEM server (Docker deployable)
│   ├── engine/            Core engine
│   │   ├── store.py       SQLite + FTS5 + sqlite-vec + knowledge graph
│   │   ├── retriever.py   Hybrid retrieval (keyword + vector + rerank + time decay)
│   │   ├── embeddings.py  bge-m3 / rerankv2m3 API client
│   │   └── reflect.py     Reflection engine (LLM clustering → mental models)
│   ├── routers/           FastAPI REST routes
│   ├── webui/             Vue3 + Element Plus + ECharts SPA
│   │   └── static/        Pre-built offline (no CDN dependency)
│   ├── main.py / config.py / middleware.py
│   ├── Dockerfile
│   └── docker-compose.yml
└── docs/
    ├── PLAN.md            Roadmap
    └── ARCH.md            Architecture design
```

### Quick Start

#### Docker (Recommended)

```bash
git clone https://github.com/icefairy/hmem_hermes_agent.git
cd hmem_hermes_agent/server

# Configure embedding API (OneAPI / OpenAI-compatible)
export EMBEDDING_BASE_URL=https://api.openai.com/v1
export EMBEDDING_API_KEY=sk-xxxx
export EMBEDDING_MODEL=text-embedding-3-small

docker compose up -d
```

Visit `http://localhost:8000` for the WebUI.

#### Local Development

```bash
cd server
pip install -r requirements.txt

EMBEDDING_BASE_URL=http://your-oneapi:3000/v1 \
EMBEDDING_API_KEY=sk-xxxx \
HMEM_DATA_DIR=/tmp/hmem \
  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HMEM_API_KEY` | `""` | API authentication key |
| `HMEM_DATA_DIR` | `/data/hmem` | SQLite data directory |
| `EMBEDDING_BASE_URL` | `""` | Embedding / rerank / LLM API base URL |
| `EMBEDDING_API_KEY` | `""` | Embedding API key |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding model |
| `EMBEDDING_DIM` | `1024` | Embedding dimension |
| `RERANK_MODEL` | `rerankv2m3` | Rerank model |
| `REFLECT_INTERVAL` | `3600` | Auto-reflection interval (seconds) |
| `REFLECT_MIN_OBSERVATIONS` | `3` | Min observations to trigger reflection |
| `REFLECT_MIN_EXPERIENCES` | `5` | Min experiences to trigger insight |
| `REFLECT_MIN_INSIGHTS` | `2` | Min insights to trigger mental model |
| `REFLECT_MODEL` | `deepseek-v4-flash` | LLM model for reflection |

### API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/memories` | Write a memory |
| GET | `/api/v1/memories` | Paginated list |
| GET | `/api/v1/memories/:id` | Single memory detail |
| DELETE | `/api/v1/memories/:id` | Delete memory |
| POST | `/api/v1/search` | Hybrid search |
| GET | `/api/v1/stats` | Statistics |
| GET | `/api/v1/graph` | Knowledge graph data |
| POST | `/api/v1/reflect` | Trigger reflection manually |
| GET | `/api/v1/mental-models` | List mental models |
| GET | `/api/v1/namespaces` | List all namespaces |
| GET | `/api/v1/logs` | Operation logs |

All requests require `Authorization: Bearer <key>` header.

### Hermes Agent Integration

```bash
# Install plugin
ln -sf /path/to/hmem/hermes-plugin ~/.hermes/plugins/hmem

# Verify
hermes plugins list
# Output: hmem  ✔  HMEM hybrid memory
```

```yaml
# ~/.hermes/config.yaml
plugins:
  hmem:
    api_url: http://localhost:8000
    api_key: sk-xxx              # Same as HMEM_API_KEY
    namespace: my-agent          # Same namespace = shared memory
```

Multiple Hermes agents with the same `namespace` share memory and mental models. Different `namespace` → fully isolated.

**Available tools**: `hmem_write`, `hmem_search`, `hmem_list`, `hmem_delete`, `hmem_stats`

### Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Storage | **SQLite + sqlite-vec** | Zero dependency, single file, fast at scale |
| Chinese NLP | **jieba** | Most mature Chinese tokenization |
| Embedding | **bge-m3** | 1024-dim, multilingual, open-source SOTA |
| Rerank | **rerankv2m3** | Cross-encoder deep ranking |
| Backend | **FastAPI + uvicorn** | Async, auto-docs |
| Frontend | **Vue3 + Element Plus + ECharts** | Dark theme, force-directed graph |
| Container | **Docker + docker compose** | One-command deploy |

### License

MIT

---

<h2 id="中文版">中文版</h2>

HMEM（Hybrid Memory）是一个**仿生记忆引擎**，专为 AI Agent 设计。受 [Hindsight](https://github.com/vectorize-io/hindsight) 启发，它不仅仅是记住事实——更重要的是**从经验中抽象出模式，形成可迁移的心智模型**。

SQLite + sqlite-vec + jieba + bge-m3 驱动，零外部依赖，一个 Docker 容器搞定。

### ✨ 核心竞争力

#### 🧠 真正的学习能力，而不仅仅是记忆

大多数记忆系统只是"信息检索器"——给什么存什么，查什么给什么。HMEM 不同。

| 能力 | HMEM | 传统 RAG | 向量数据库 |
|------|------|----------|-----------|
| 关键词全文检索 | ✅ FTS5 + jieba 中文分词 | ✅ | ❌ |
| 语义向量搜索 | ✅ bge-m3 1024维 | ✅ | ✅ |
| 交叉编码器重排 | ✅ rerankv2m3 深度排序 | ❌ | ❌ |
| **时间衰减权重** | ✅ 近期记忆自动加权 | ❌ | ❌ |
| **知识图谱关系** | ✅ memory_edges 关联网络 | ❌ | ❌ |
| **反思→心智模型** | ✅ LLM 聚类分析 → 抽象模式 | ❌ | ❌ |
| **可视化管理** | ✅ Vue3 WebUI（图谱/搜索/反思） | ❌ | ❌ |

#### 🔥 Reflect 引擎：从经验到智慧的跃迁

这是 HMEM 与其他所有记忆方案的本质区别。

```
写入记忆                    反思循环                    检索时
━━━━━━━━━━━━━━━            ━━━━━━━━━━━━━━            ━━━━━━━━━━━━━━
用户: "用FastAPI写了     ┌─ 累积 50+ 条经验 ─┐      不仅要查事实，
      一个文件上传接口    │                     │      还要匹配模式：
用户: "PostgreSQL       │  LLM 聚类分析      │      "这个用户偏好
      查询很慢……"         │                     │       详细的方案说明"
用户: "用户反馈代码      │  └─ 心智模型 ──────┘
      注释不够详细"      →    "该用户偏好详细
用户: "部署到K8s          │   的技术方案说明，
      遇到网络问题……"      │   包含实际代码示例"      回答自动调整风格
                          │                     │
                        心智模型指导后续行为 ←──┘
```

**这意味着**：Agent 用久了会越来越懂你。不是靠 prompt 预设，而是真正从交互中学习。

#### 🚀 混合检索：四维并行

```
关键词检索 (FTS5)  ────→ 精确匹配专业术语 "pgvector"
向量语义 (bge-m3)  ────→ "数据库慢" → 找到 "PostgreSQL 查询优化"
图关系 (memory_edges) ──→ "上次那个问题" → 找到关联的调试记录
时间衰减             ────→ 近期记忆权重更高
         │
         ▼
   交叉编码器重排  ──────→ 深度语义理解的排序
         │
         ▼
   最终 Top-K 结果
```

#### 🏗️ 物理级隔离，天生支持多 Agent

每个 **namespace** 对应一个独立的 SQLite 文件。

```
/data/hmem/
├── team-alpha.db       ← code-review / debug-assistant 共享记忆
├── personal-duck.db    ← 个人助手私有
└── agent-pilot.db      ← 另一个独立 Agent
```

同 namespace → 共享记忆和心智模型，不同 namespace → 物理隔离互不干扰。

#### 📦 零外部依赖

| 依赖 | HMEM | PostgreSQL 方案 |
|------|------|----------------|
| 数据库 | **SQLite**（单文件） | PostgreSQL 集群 |
| 向量引擎 | **sqlite-vec**（内嵌） | pgvector 插件 |
| 中间件 | **无** | Redis / 消息队列 |
| 部署 | **docker compose up** | 至少 3 个容器 |
| 备份 | **cp 一个文件** | pg_dump 全量 |

**一个 Docker 容器，开箱即用。**

#### 🖥️ 可视化大脑：WebUI 管理界面

内置 Vue3 + ECharts 管理界面。

- **📊 概览**：记忆数量、类型分布、嵌入状态一目了然
- **🔍 搜索**：混合检索，每条结果标注来源（关键词/向量/重排）
- **🕸️ 知识图谱**：力导向图展示记忆之间的关联网络
- **🧠 反思**：手动触发反思，查看心智模型列表
- **⚙️ 设置**：配置阈值、夜间定时反思窗口

### 架构

```
hmem/
├── hermes-plugin/         Hermes Agent 插件（~300 行 HTTP API 客户端）
│   └── __init__.py        只需配置 api_url + api_key + namespace
├── server/                HMEM 服务端（Docker 部署）
│   ├── engine/            记忆引擎核心
│   │   ├── store.py       SQLite + FTS5 + sqlite-vec + 知识图谱
│   │   ├── retriever.py   混合检索（关键词 + 向量 + 重排 + 时间衰减）
│   │   ├── embeddings.py  bge-m3 / rerankv2m3 API 客户端
│   │   └── reflect.py     反思引擎（LLM 聚类→心智模型抽象）
│   ├── routers/           FastAPI REST 路由
│   ├── webui/             Vue3 + Element Plus + ECharts SPA
│   │   └── static/        预构建离线（无 CDN 依赖）
│   ├── main.py / config.py / middleware.py
│   ├── Dockerfile
│   └── docker-compose.yml
└── docs/
```

### 快速启动

#### Docker（推荐）

```bash
git clone https://github.com/icefairy/hmem_hermes_agent.git
cd hmem_hermes_agent/server

# 配置嵌入 API（OneAPI / OpenAI 兼容）
export EMBEDDING_BASE_URL=https://api.openai.com/v1
export EMBEDDING_API_KEY=sk-xxxx
export EMBEDDING_MODEL=text-embedding-3-small

docker compose up -d
```

访问 `http://localhost:8000` 即可看到 WebUI。

#### 本地开发

```bash
cd server
pip install -r requirements.txt

EMBEDDING_BASE_URL=http://your-oneapi:3000/v1 \
EMBEDDING_API_KEY=sk-xxxx \
HMEM_DATA_DIR=/tmp/hmem \
  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HMEM_API_KEY` | `""` | API 认证密钥 |
| `HMEM_DATA_DIR` | `/data/hmem` | SQLite 数据目录 |
| `EMBEDDING_BASE_URL` | `""` | 嵌入/重排/LLM API 地址 |
| `EMBEDDING_API_KEY` | `""` | 嵌入 API 密钥 |
| `EMBEDDING_MODEL` | `bge-m3` | 嵌入模型 |
| `EMBEDDING_DIM` | `1024` | 嵌入维度 |
| `RERANK_MODEL` | `rerankv2m3` | 重排模型 |
| `REFLECT_INTERVAL` | `3600` | 自动反思间隔（秒） |
| `REFLECT_MIN_OBSERVATIONS` | `3` | 最少观察数触发反思 |
| `REFLECT_MIN_EXPERIENCES` | `5` | 最少经验数触发洞见 |
| `REFLECT_MIN_INSIGHTS` | `2` | 最少洞见数触发心智模型 |
| `REFLECT_MODEL` | `deepseek-v4-flash` | 反思用 LLM 模型 |

### API 参考

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 探活 |
| POST | `/api/v1/memories` | 写入记忆 |
| GET | `/api/v1/memories` | 分页列表 |
| GET | `/api/v1/memories/:id` | 单条详情 |
| DELETE | `/api/v1/memories/:id` | 删除 |
| POST | `/api/v1/search` | 混合检索 |
| GET | `/api/v1/stats` | 统计 |
| GET | `/api/v1/graph` | 知识图谱数据 |
| POST | `/api/v1/reflect` | 手动触发反思 |
| GET | `/api/v1/mental-models` | 心智模型列表 |
| GET | `/api/v1/namespaces` | 所有命名空间 |
| GET | `/api/v1/logs` | 操作日志 |

所有请求需携带 `Authorization: Bearer <key>` 请求头。

### Hermes Agent 集成

```bash
# 安装插件
ln -sf /path/to/hmem/hermes-plugin ~/.hermes/plugins/hmem

# 验证
hermes plugins list
# 输出: hmem  ✔  HMEM hybrid memory
```

```yaml
# ~/.hermes/config.yaml
plugins:
  hmem:
    api_url: http://localhost:8000
    api_key: sk-xxx              # 与 HMEM_API_KEY 一致
    namespace: my-agent          # 同空间 = 共享记忆
```

多个 Hermes Agent 配置相同 `namespace` → 共享记忆和心智模型。不同 `namespace` → 完全隔离。

**可用工具**：`hmem_write`, `hmem_search`, `hmem_list`, `hmem_delete`, `hmem_stats`

### 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| 存储 | **SQLite + sqlite-vec** | 零依赖，单文件，千万级够用 |
| 中文分词 | **jieba** | 最成熟的中文 NLP 库 |
| 向量模型 | **bge-m3** | 1024 维，多语言，开源 SOTA |
| 重排模型 | **rerankv2m3** | 交叉编码器深度排序 |
| 后端 | **FastAPI + uvicorn** | 异步高性能，自动文档 |
| 前端 | **Vue3 + Element Plus + ECharts** | 暗色主题，力导向图谱 |
| 容器 | **Docker + docker compose** | 一键部署 |
| Agent 集成 | **Hermes MemoryProvider** | 标准插件接口 |

### 许可

MIT

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=icefairy/hmem_hermes_agent&type=Date)](https://star-history.com/#icefairy/hmem_hermes_agent&Date)

> **HMEM：让 AI Agent 从工具进化为伙伴。**  
> **HMEM: Turning AI Agents from tools into partners.**