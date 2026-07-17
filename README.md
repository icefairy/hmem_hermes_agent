# HMEM · 混合记忆系统

> **让 AI Agent 记住每一次对话，从经验中学习，不再重复犯错。**

HMEM 是一个受 [Hindsight](https://github.com/vectorize-io/hindsight) 启发的仿生记忆引擎。它让 Hermes Agent 具备了人类般的长期学习能力——不仅仅是回忆事实，更重要的是**从经验中抽象出模式，形成可迁移的心智模型**。

SQLite + sqlite-vec + jieba + bge-m3 驱动，零外部依赖，`docker compose up` 一键启动。

---

## ✨ 核心竞争力

### 🧠 真正的学习能力，而不仅仅是记忆

大多数记忆系统只是「信息检索器」——给什么存什么，查什么给什么。HMEM 不同。

| 能力 | HMEM | 传统 RAG | 向量数据库 |
|------|------|----------|-----------|
| 关键词全文检索 | ✅ FTS5 + jieba 中文分词 | ✅ | ❌ |
| 语义向量搜索 | ✅ bge-m3 1024维 | ✅ | ✅ |
| 交叉编码器重排 | ✅ rerankv2m3 深度排序 | ❌ | ❌ |
| **时间衰减权重** | ✅ 近期记忆自动加权 | ❌ | ❌ |
| **知识图谱关系** | ✅ memory_edges 关联网络 | ❌ | ❌ |
| **反思→心智模型** | ✅ LLM 聚类分析 → 抽象模式 | ❌ | ❌ |
| **可视化管理** | ✅ Vue3 WebUI（图谱/搜索/反思） | ❌ | ❌ |

### 🔥 Reflect 引擎：从经验到智慧的跃迁

这是 HMEM 与所有其他记忆方案的本质区别。

```
写入记忆                    反思循环                    检索时
━━━━━━━━━━━━━━            ━━━━━━━━━━━━━━            ━━━━━━━━━━━━━━
用户: "用FastAPI写了     ┌─ 累积 50+ 条经验 ─┐      不仅要查事实，
     一个文件上传接口     │                     │      还要匹配模式：
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

### 🚀 混合检索：四维并行，一个不落

```
关键词检索 (FTS5)  ────→ 精确匹配专业术语 "pgvector"
向量语义 (bge-m3)  ────→ "数据库慢" → 找到 "PostgreSQL 查询优化"
图关系 (memory_edges) ──→ "上次那个问题" → 找到关联的调试记录    [预留]
时间衰减             ────→ 近期的记忆权重更高，但旧知识不被遗忘
         │
         ▼
   交叉编码器重排  ──────→ 不是简单分数加权，而是深度语义理解的排序
         │
         ▼
   最终 Top-K 结果
```

### 🏗️ 物理级隔离，天生支持多 Agent

每个 **namespace** 对应一个独立的 SQLite 文件。

```
/data/hmem/
├── team-alpha.db       ← code-review / debug-assistant 共享记忆
├── personal-duck.db    ← 个人助手私有
└── agent-pilot.db      ← 另一个独立 Agent

两个 Agent 配置同一个 namespace → 共享记忆
不同 namespace → 物理隔离，互不干扰
```

没有复杂的权限模型，不依赖 PostgreSQL schema，文件级别隔离，干净利落。

### 📦 零外部依赖

| 依赖 | HMEM | PostgreSQL 方案 |
|------|------|----------------|
| 数据库 | **SQLite**（单文件） | PostgreSQL 集群 |
| 向量引擎 | **sqlite-vec**（内嵌） | pgvector 插件 |
| 中间件 | **无** | Redis / 消息队列 |
| 部署 | **docker compose up** | 至少 3 个容器 |
| 备份 | **cp 一个文件** | pg_dump 全量 |

**一个 Docker 容器，开箱即用。** 没有繁重的运维负担，适合从个人项目到团队协作的任何场景。

### 🖥️ 可视化大脑：WebUI 管理界面

HMEM 内置 Vue3 + ECharts 管理界面，让你的记忆系统变得可视化。

- **📊 概览**：记忆数量、类型分布、嵌入状态一目了然
- **🔍 搜索**：混合检索，每条结果标注来源（关键词/向量/重排）
- **🕸️ 知识图谱**：力导向图展示记忆之间的关联网络，心智模型高亮显示
- **🧠 反思**：手动触发反思，查看心智模型列表和支撑经验

---

## 项目结构

```
hmem/
├── hermes-plugin/         Hermes Agent 插件（HTTP API 客户端，~300 行）
│   └── __init__.py        只需配置 api_url + api_key + namespace
├── server/                HMEM 服务端（Docker 部署）
│   ├── engine/            记忆引擎核心
│   │   ├── store.py       SQLite + FTS5 + sqlite-vec + 知识图谱
│   │   ├── retriever.py   混合检索（关键词 + 向量 + 重排 + 时间衰减）
│   │   ├── embeddings.py  bge-m3 / rerankv2m3 API 客户端
│   │   └── reflect.py     反思引擎（LLM 聚类→心智模型抽象）
│   ├── routers/           FastAPI REST 路由
│   ├── webui/             Vue3 + Element Plus + ECharts SPA
│   ├── main.py / config.py / middleware.py
│   ├── Dockerfile
│   └── docker-compose.yml
└── docs/
    ├── PLAN.md            升级规划
    └── ARCH.md            架构设计
```

---

## 快速启动

### 方式一：Docker 部署（推荐）

```bash
cd server

# 构建 WebUI 前端
cd webui && pnpm install && pnpm build && cd ..

# 启动（需先配置 EMBEDDING_API_KEY）
export EMBEDDING_BASE_URL=http://your-oneapi:3000/v1
export EMBEDDING_API_KEY=sk-xxxx
docker compose up -d
```

访问 `http://localhost:8000` 即可看到 WebUI。

### 方式二：本地开发

```bash
cd server
pip install -r requirements.txt

EMBEDDING_BASE_URL=http://your-oneapi:3000/v1 \
EMBEDDING_API_KEY=sk-xxxx \
HMEM_DATA_DIR=/tmp/hmem \
  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Hermes Agent 集成

### 安装插件

```bash
# 方式一：从本地项目目录安装（推荐，开发和调试）
# 确保 hermes-plugin/ 与 README.md 同级
ln -sf /path/to/hmem/hermes-plugin ~/.hermes/plugins/hmem

# 方式二：从 Git 仓库安装
git clone http://192.168.1.10:3000/icefairy/hmem.git /tmp/hmem
ln -sf /tmp/hmem/hermes-plugin ~/.hermes/plugins/hmem

# 方式三：手动放置
cp -r hermes-plugin ~/.hermes/plugins/hmem

# 验证安装（插件名 hmem 会自动扫描）
hermes plugins list
# 应输出: hmem  ✔  HMEM hybrid memory
```

### 配置

```yaml
# ~/.hermes/config.yaml
plugins:
  hmem:
    api_url: http://hmem:8000   # HMEM 服务地址
    api_key: sk-xxx              # 与 HMEM_API_KEY 一致
    namespace: team-alpha        # 记住：同空间 = 共享记忆
```

多个 Hermes Agent 配置相同的 `namespace` → 共享记忆和心智模型。配置不同的 `namespace` → 完全隔离。

### Hermes Tools

| Tool | 功能 |
|------|------|
| `hmem_write` | 写入记忆（支持 content / namespace / mem_action） |
| `hmem_search` | 混合检索（FTS5 + 向量 + rerank + 时间衰减） |
| `hmem_list` | 最近记忆列表 |
| `hmem_delete` | 删除记忆 |
| `hmem_stats` | 统计信息 |

---

## API 参考

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 探活 |
| POST | `/api/v1/memories` | 写入记忆 |
| GET | `/api/v1/memories?namespace=x` | 分页列表 |
| GET | `/api/v1/memories/:id?namespace=x` | 单条详情 |
| DELETE | `/api/v1/memories/:id?namespace=x` | 删除 |
| POST | `/api/v1/search` | 混合检索 |
| GET | `/api/v1/stats?namespace=x` | 统计 |
| GET | `/api/v1/graph?namespace=x` | 知识图谱（力导向图数据） |
| POST | `/api/v1/reflect` | 手动触发反思 |
| GET | `/api/v1/mental-models?namespace=x` | 心智模型列表 |

所有请求需携带 `Authorization: Bearer <api_key>`。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HMEM_API_KEY` | `""` | API 认证密钥 |
| `HMEM_DATA_DIR` | `/data/hmem` | SQLite 数据目录 |
| `EMBEDDING_BASE_URL` | `""` | 嵌入/重排/LLM API 地址 |
| `EMBEDDING_API_KEY` | `""` | 嵌入 API 密钥 |
| `EMBEDDING_MODEL` | `bge-m3` | 嵌入模型 |
| `RERANK_MODEL` | `rerankv2m3` | 重排模型 |
| `REFLECT_INTERVAL` | `3600` | 反思间隔（秒） |
| `REFLECT_MIN_EXPERIENCES` | `50` | 最少经验数触发反思 |
| `REFLECT_MODEL` | `deepseek-v4-flash` | 反思用 LLM 模型 |

---

## 技术栈

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

---

## Git

```bash
git remote add origin http://192.168.1.10:3000/icefairy/hmem.git
git push origin master
```

---

> **HMEM：让 AI Agent 从工具进化为伙伴。**