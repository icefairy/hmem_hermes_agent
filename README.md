# HMEM · 混合记忆系统

SQLite + sqlite-vec + jieba + bge-m3 驱动的混合记忆引擎，对标 [Hindsight](https://github.com/vectorize-io/hindsight) 仿生记忆架构。

每个 **namespace** 一个独立的 SQLite 文件，多个 agent 配置相同 namespace 即共享记忆。

---

## 项目结构

```
hmem/
├── hermes-plugin/         Hermes Agent 插件（HTTP API 客户端，~300 行）
│   └── __init__.py        只需配置 api_url + api_key + namespace
├── server/                HMEM 服务端
│   ├── engine/            记忆引擎核心
│   │   ├── store.py       SQLite + FTS5 + sqlite-vec + 知识图谱
│   │   ├── retriever.py   混合检索（关键词 + 向量 + 重排）
│   │   ├── embeddings.py  bge-m3 / rerankv2m3 API 客户端
│   │   └── reflect.py     反思引擎（经验 → 心智模型抽象）
│   ├── routers/           FastAPI REST 路由
│   │   ├── memories.py    CRUD
│   │   ├── search.py      混合检索
│   │   ├── stats.py       统计
│   │   ├── graph.py       知识图谱
│   │   ├── reflect.py     触发反思
│   │   └── mental_models.py 心智模型
│   ├── webui/             Vue3 + Element Plus SPA
│   │   └── src/App.vue    概览/搜索/图谱/反思页面
│   ├── main.py            FastAPI 入口
│   ├── config.py          环境变量配置
│   ├── middleware.py       Bearer token 认证
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
└── docs/
    ├── PLAN.md            升级规划（对标 Hindsight）
    └── ARCH.md            架构设计
```

---

## 快速启动

### 方式一：本地开发

```bash
# 1. 安装依赖
cd server
pip install -r requirements.txt

# 2. 启动 HMEM Server
HMEM_DATA_DIR=/tmp/hmem \
EMBEDDING_BASE_URL=http://localhost:3000/v1 \
EMBEDDING_API_KEY=sk-KMNYyAXDh1REPLACED_HMEM_KEY \
EMBEDDING_MODEL=bge-m3 \
RERANK_MODEL=rerankv2m3 \
  python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 3. 另一个终端：开发 WebUI（可选）
cd server/webui
pnpm install
pnpm dev   # 默认 http://localhost:5173
```

### 方式二：Docker 部署（生产推荐）

```bash
cd server

# 先构建 WebUI
cd webui && pnpm install && pnpm build && cd ..

# 启动
HMEM_API_KEY=my-secret-key \
EMBEDDING_BASE_URL=http://localhost:3000/v1 \
EMBEDDING_API_KEY=sk-KMNYyAXDh1REPLACED_HMEM_KEY \
  docker compose up -d
```

访问 `http://localhost:8000` 即可看到 WebUI 管理界面。

---

## Hermes Agent 集成

### 安装插件

```bash
# 将 hmem 插件注册到 Hermes
cp -r hermes-plugin ~/.hermes/plugins/hmem

# 配置
hermes config set plugins.hmem.api_url http://hmem:8000
hermes config set plugins.hmem.api_key my-secret-key
hermes config set plugins.hmem.namespace team-alpha
```

### 配置示例（config.yaml）

```yaml
plugins:
  hmem:
    api_url: http://hmem-server:8000    # HMEM 服务地址
    api_key: my-secret-key               # 与 HMEM_API_KEY 一致
    namespace: team-alpha                # 记忆命名空间
```

**注意**：多个 Hermes Agent 配置相同的 namespace → 共享记忆；不同的 namespace → 物理隔离（独立 db 文件）。

### Hermes Tools

| Tool | 功能 |
|------|------|
| `hmem_write` | 写入记忆（支持 content / namespace / mem_action 字段） |
| `hmem_search` | 混合检索（关键词 + 向量语义 + 重排） |
| `hmem_list` | 最近记忆列表 |
| `hmem_delete` | 删除记忆 |
| `hmem_stats` | 统计信息 |

---

## API 参考

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 探活 |
| POST | `/api/v1/memories` | 写入记忆（`namespace` 决定 db 文件） |
| GET | `/api/v1/memories?namespace=x` | 分页列表 |
| GET | `/api/v1/memories/:id?namespace=x` | 单条详情 |
| DELETE | `/api/v1/memories/:id?namespace=x` | 删除 |
| POST | `/api/v1/search` | 混合检索 |
| GET | `/api/v1/stats?namespace=x` | 统计 |
| GET | `/api/v1/graph?namespace=x` | 知识图谱数据（力导向图） |
| POST | `/api/v1/reflect` | 手动触发反思 |
| GET | `/api/v1/mental-models?namespace=x` | 心智模型列表 |

所有请求需携带 `Authorization: Bearer <api_key>`。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HMEM_API_KEY` | `""` | API 认证密钥（空=无认证） |
| `HMEM_DATA_DIR` | `/data/hmem` | SQLite 数据目录 |
| `HMEM_DEBUG` | `false` | 调试模式 |
| `EMBEDDING_BASE_URL` | `""` | bge-m3 向量 API 地址 |
| `EMBEDDING_API_KEY` | `""` | 向量 API 密钥 |
| `EMBEDDING_MODEL` | `bge-m3` | 嵌入模型名 |
| `RERANK_MODEL` | `rerankv2m3` | 重排模型名 |
| `EMBEDDING_DIM` | `1024` | 向量维度 |
| `REFLECT_INTERVAL` | `3600` | 反思间隔（秒） |
| `REFLECT_MIN_EXPERIENCES` | `50` | 最少经验数才触发反思 |

---

## Git

```bash
# 远端
git remote add origin http://192.168.1.10:3000/icefairy/hmem.git
git remote add gitea http://172.16.11.6:10001/qinlingbing/hmem.git

# 推送
git push origin master
git push gitea master
```