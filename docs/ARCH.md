# HMEM · 混合记忆系统架构

## 核心思想

**轻量插件 + 独立引擎**：Hermes 插件只做 API 客户端，记忆引擎核心和 WebUI 放在一起，用 Docker 一键部署。

```
┌─────────────────────────────────────────────────────────────┐
│                    Hermes Agent                              │
│  ┌──────────────────────────────────────────────┐           │
│  │ hmem-plugin (轻量 API 客户端)                  │           │
│  │  └─ 通过 http://hmem:8000 + api_key 调用       │           │
│  └──────────┬───────────────────────────────────┘           │
└─────────────┼───────────────────────────────────────────────┘
              │ HTTP (OpenAI-compatible API)
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    HMEM Server (Docker)                      │
│                                                              │
│  ┌──────────────────────────┐  ┌────────────────────────┐   │
│  │    FastAPI Server         │  │    WebUI (Vue3 SPA)    │   │
│  │  ┌──────────────────────┐ │  │  ┌──────────────────┐ │   │
│  │  │ /api/v1/memories     │ │  │  │ Dashboard        │ │   │
│  │  │ /api/v1/search       │ │  │  │ Search           │ │   │
│  │  │ /api/v1/reflect      │ │  │  │ Knowledge Graph  │ │   │
│  │  │ /api/v1/graph        │ │  │  │ Reflect History  │ │   │
│  │  │ /api/v1/mental-models│ │  │  └──────────────────┘ │   │
│  │  └──────────────────────┘ │  └────────────────────────┘   │
│  └──────────┬───────────────┘                                │
│             │                                                │
│  ┌──────────▼──────────────────────────────────────┐        │
│  │         HMEM Engine (核心模块)                    │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │        │
│  │  │ store.py │ │retriever │ │embeddings│        │        │
│  │  │ (SQLite) │ │ .py      │ │ .py      │        │        │
│  │  │  +FTS5   │ │(四维检索) │ │(bge-m3   │        │        │
│  │  │  +vec0   │ │          │ │ rerank)  │        │        │
│  │  └──────────┘ └──────────┘ └──────────┘        │        │
│  │  ┌──────────────────────────────────────┐      │        │
│  │  │ reflect.py (Reflect 引擎)            │      │        │
│  │  │  定时分析经验 → 聚类 → 心智模型抽象       │      │        │
│  │  └──────────────────────────────────────┘      │        │
│  └─────────────────────────────────────────────────┘       │
│                                                              │
│  存储：SQLite (单文件，零依赖)                                  │
│  └── /data/hmem/hybrid_memory.db                             │
└─────────────────────────────────────────────────────────────┘
```

## API 设计

### 认证
所有请求携带 `Authorization: Bearer <api_key>`，api_key 通过环境变量 `HMEM_API_KEY` 配置。

### 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/health` | 探活 |
| POST | `/api/v1/memories` | 写入记忆 |
| POST | `/api/v1/search` | 混合检索 |
| GET | `/api/v1/memories` | 分页列表 |
| GET | `/api/v1/memories/:id` | 单条详情 |
| DELETE | `/api/v1/memories/:id` | 删除 |
| GET | `/api/v1/stats` | 统计 |
| GET | `/api/v1/graph` | 记忆关系图数据 |
| POST | `/api/v1/reflect` | 手动触发反思 |
| GET | `/api/v1/mental-models` | 心智模型列表 |

## Hermes 插件 (hmem-plugin)

轻量实现，只做 API 转发：

```python
# hmem-plugin/__init__.py
class HmemMemoryProvider(MemoryProvider):
    def __init__(self, config):
        self._api_url = config.get("api_url", "http://localhost:8000")
        self._api_key = config.get("api_key", "")
    
    def _call(self, method, path, body=None):
        headers = {"Authorization": f"Bearer {self._api_key}"}
        # httpx 调用...
    
    def handle_tool_call(self, name, args):
        # 直接透传 JSON → POST /api/v1/...
        return self._call("POST", f"/api/v1/{name}", body=args)
```

Hermes profile 配置：

```yaml
plugins:
  hmem:
    api_url: http://hmem-server:8000
    api_key: my-secret-key
```

## 目录结构

```
hmem/
├── server/                         # HMEM Server (Docker 部署)
│   ├── main.py                     # FastAPI 入口
│   ├── config.py                   # 配置（env 驱动）
│   ├── middleware.py               # 认证中间件
│   ├── engine/                     # 记忆引擎核心
│   │   ├── __init__.py
│   │   ├── store.py                # SQLite + FTS5 + vec0
│   │   ├── retriever.py            # 四维并行检索
│   │   ├── embeddings.py           # bge-m3 / rerank 客户端
│   │   └── reflect.py              # 反思引擎
│   ├── webui/                      # Vue3 前端
│   │   ├── index.html
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   └── src/
│   │       ├── main.ts
│   │       ├── App.vue
│   │       └── views/
│   │           ├── Dashboard.vue
│   │           ├── Search.vue
│   │           ├── KnowledgeGraph.vue
│   │           └── Reflect.vue
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── hermes-plugin/                  # Hermes 轻量插件
│   └── __init__.py                 # 仅 API 客户端，~200 行
│
├── docs/
│   ├── PLAN.md
│   └── ADR-001-architecture.md
│
├── .gitignore
└── README.md
```

## Docker 部署

```yaml
# docker-compose.yml
services:
  hmem:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - hmem-data:/data/hmem
    environment:
      HMEM_API_KEY: ${HMEM_API_KEY:-changeme}
      EMBEDDING_BASE_URL: http://oneapi:3000/v1
      EMBEDDING_API_KEY: ${EMBEDDING_API_KEY}
      EMBEDDING_MODEL: bge-m3
      RERANK_MODEL: rerankv2m3
      REFLECT_INTERVAL: 3600
      REFLECT_MIN_EXPERIENCES: 50

volumes:
  hmem-data:
```

## 与 Hindsight 的对比

| 维度 | Hindsight | HMEM |
|------|-----------|------|
| 存储引擎 | PostgreSQL + 向量库 | **SQLite + sqlite-vec** |
| 部署依赖 | 需要 PG 服务、向量库集群 | **零依赖，单文件** |
| API | RESTful + Python SDK | **OpenAI 兼容 + REST** |
| 插件集成 | HindsightWrapper 包装 | **Hermes MemoryProvider** |
| 反思引擎 | Rust 高性能 | Python (可配置间隔) |
| 可视化 | 无原生 WebUI | **内置 Vue3 SPA** |
| 知识图谱 | 有 | **有（ECharts 力导向图）** |