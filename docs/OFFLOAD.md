# HMEM 上下文卸载（Context Offloading）

把长会话/长任务中的厚重内容（工具调用结果、大段文本、日志）卸载到外部存储，
上下文只保留**轻量摘要 + 索引**，需要时按 `node_id` 找回完整原文。

原理移植自 TencentDB Agent Memory（`MemoryCore/src/offload/`），三层结构落地为：

| 层 | TencentDB | HMEM 实现 |
|----|-----------|-----------|
| 底层 完整原文存档 | `refs/*.md` 文件系统 | `{HMEM_DATA_DIR}/offload/{namespace}/{session_key}/refs/{node_id}.md` |
| 中层 摘要索引 | `offload-*.jsonl` | SQLite `offload_records` 表（summary / node_id / refs_path / content_hash） |
| 高层 任务画布 | Mermaid 画布 | `GET /offload/canvas/{key}` 生成 flowchart（节点=动作，边=depends_on，click 可下钻） |

## 设计要点

- **摘要必须短**：自动摘要压缩为单行、限长 120 字符（`DEFAULT_SUMMARY_CHARS`），适合直接注入上下文
- **100% 可找回**：`node_id → refs 文件` 链路完整，写入带 `content_hash`（sha256 前 16 位），读出可校验
- **原文原子写**：tmp 文件 + `os.replace`，不会出现半截文件
- **路径穿越防护**：`session_key`/`node_id` 白名单清洗（非法字符替换为 `_`），refs 解析校验不越界
- **软删除**：默认软删（deleted=1，refs 目录移入 `.trash/`，增量数据保留），`hard=true` 才物理删除
- **namespace 隔离**：SQLite 沿用现有分库（`{HMEM_DATA_DIR}/{namespace}.db`），offload 文件按 `offload/{namespace}/` 分目录
- **auth**：复用全局 `AuthMiddleware`（`Authorization: Bearer <key>`）

## 存储结构

```
{HMEM_DATA_DIR}/offload/{namespace}/{session_key}/
    refs/{node_id}.md          # 完整原文（纯原文，无头）
    ...                        # 摘要记录在 SQLite offload_records 表
    .trash/                    # 软删除的会话目录（增量数据保留）
```

SQLite `offload_records` 表字段：`session_key, node_id, summary, content_type, refs_path, meta(JSON), content_hash, deleted, created_at, updated_at`，`UNIQUE(session_key, node_id)`（重复 put = 覆盖更新）。

## API

Base: `http://<host>:8090/api/v1` · 所有请求带 `Authorization: Bearer <HMEM_API_KEY>`

### 1. 创建/获取会话卸载空间（幂等）

```bash
curl -s -X POST http://127.0.0.1:8090/api/v1/offload/session \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"session_key": "task-20260807", "namespace": "default"}'
```

```json
{"session_key":"task-20260807","meta":{},"deleted":false,"created_at":"...","updated_at":"...","record_count":0,"refs_dir":"/data/hmem-data/offload/default/task-20260807/refs"}
```

### 2. 卸载一条内容

```bash
curl -s -X POST http://127.0.0.1:8090/api/v1/offload/put \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
    "session_key": "task-20260807",
    "node_id": "step-3",                     # 可选；不传自动生成 node_1, node_2...
    "content": "工具返回的大段 JSON 结果……",  # 完整原文，落 refs 文件
    "summary": "查询订单 #1024 状态：已发货",   # 可选；不传自动生成一行摘要
    "content_type": "tool_result",
    "meta": {"depends_on": "step-2", "request_id": "r-42"}
  }'
```

```json
{"session_key":"task-20260807","node_id":"step-3","summary":"查询订单 #1024 状态：已发货","content_type":"tool_result","refs_path":"refs/step-3.md","content_hash":"9f2c…","created_at":"..."}
```

### 3. 找回完整原文

```bash
curl -s "http://127.0.0.1:8090/api/v1/offload/get?session_key=task-20260807&node_id=step-3" \
  -H "Authorization: Bearer $KEY"
```

```json
{"id":1,"session_key":"task-20260807","node_id":"step-3","summary":"…","content_type":"tool_result","refs_path":"refs/step-3.md","meta":{...},"content_hash":"…","deleted":false,"created_at":"...","updated_at":"...","content":"工具返回的大段 JSON 结果……"}
```

### 4. 会话索引（注入上下文的摘要列表）

```bash
curl -s "http://127.0.0.1:8090/api/v1/offload/session/task-20260807?include_deleted=false" \
  -H "Authorization: Bearer $KEY"
```

```json
{"session_key":"task-20260807","count":2,"limit":10000,"offset":0,"records":[{"id":1,"node_id":"step-2","summary":"…","refs_path":"refs/step-2.md",...},{"id":2,"node_id":"step-3","summary":"…",...}]}
```

> 索引**不含原文**（`content` 字段不会出现在 records 里），正是为省 token 设计的注入形态。

### 5. Mermaid 任务画布

```bash
curl -s "http://127.0.0.1:8090/api/v1/offload/canvas/task-20260807" -H "Authorization: Bearer $KEY"
```

```json
{"session_key":"task-20260807","namespace":"default","mermaid":"flowchart TD\n    step-2[\"...\"]\n    step-3[\"...\"]\n    step-2 --> step-3\n    click step-2 \"refs/step-2.md\" \"查看原文\"\n"}
```

### 6. 清理会话

```bash
# 软删除（默认）：deleted=1 + refs 移入 .trash，增量数据保留
curl -s -X DELETE "http://127.0.0.1:8090/api/v1/offload/session/task-20260807" -H "Authorization: Bearer $KEY"
# → {"mode":"soft","count":2}

# 硬删除：物理删除记录和原文文件
curl -s -X DELETE "http://127.0.0.1:8090/api/v1/offload/session/task-20260807?hard=true" -H "Authorization: Bearer $KEY"
# → {"mode":"hard","count":2}
```

## 与 Hermes Agent 配合的使用姿势

1. 会话接近上下文上限时，把最近几轮的工具调用结果逐个 `PUT /offload/put`（node_id 用 tool_call_id）
2. 用 `GET /offload/session/{key}` 拿摘要列表拼进上下文（替代原文，省 ~61% token）
3. 模型需要原文时，按摘要里的 node_id `GET /offload/get` 精准找回
4. 任务收尾：默认软删保留增量，确认无用后再 `hard=true` 硬删

## 测试

```bash
cd /data/codes/hmem && venv/bin/python -m pytest server/tests/test_offload.py -v
# 14 passed：put→get 找回、自动摘要/自动 node_id、upsert、索引、软/硬删、画布、namespace 隔离、auth、路径穿越
```
