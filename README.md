# HMEM · 混合记忆系统

SQLite + sqlite-vec + jieba + bge-m3 驱动的 Hermes Agent 混合记忆插件，对标 Hindsight 仿生记忆架构。

## 目录结构

```
hermes-agent/  →  Hermes Agent 插件（MemoryProvider 实现）
webui/         →  Vue3 + Element Plus 可视化前端 + FastAPI 后端
docs/          →  架构文档、规划、ADR
```

## 快速上手

```bash
# 激活到 Hermes 默认 profile
cp -r hermes-agent/plugins/hybrid-memory ~/.hermes/plugins/
hermes config set memory.provider hybrid-memory

# 启动 WebUI
cd webui && pnpm install && pnpm dev
```