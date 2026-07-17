"""HMEM Server — FastAPI 入口"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from config import Settings
from middleware import AuthMiddleware
from routers import memories, search, stats, graph, reflect, mental_models
from engine.store import HybridMemoryStore

logger = logging.getLogger(__name__)

SPA_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>HMEM · 混合记忆系统</title>
  <link rel="stylesheet" href="/static/element-plus.css">
  <style>
    body { margin:0; background:#0d1117; color:#e6edf3; font-family:system-ui,sans-serif; }
    .el-header { border-bottom:1px solid #30363d; background:#161b22; display:flex; align-items:center; justify-content:space-between; padding:0 24px; height:56px; }
    .el-main { padding:24px; background:#0d1117; }
    .el-card { background:#161b22; border-color:#30363d; color:#e6edf3; margin-bottom:16px; }
    .el-table { --el-table-bg-color:#161b22; --el-table-tr-bg-color:#161b22; --el-table-header-bg-color:#21262d; --el-table-border-color:#30363d; --el-table-text-color:#e6edf3; --el-table-header-text-color:#e6edf3; --el-table-row-hover-bg-color:#1c2128; }
    .el-input { --el-input-bg-color:#0d1117; --el-input-border-color:#30363d; --el-input-text-color:#e6edf3; --el-input-hover-border-color:#409eff; }
    .el-dialog { --el-dialog-bg-color:#161b22; --el-dialog-title-text-color:#e6edf3; }
  </style>
</head>
<body>
<div id="app">
<el-header style="border-bottom:1px solid #30363d;background:#161b22;display:flex;align-items:center;justify-content:space-between;padding:0 24px;height:56px">
  <h2 style="margin:0;font-size:18px">🧠 HMEM</h2>
  <div>
    <el-input v-model="namespace" placeholder="ns" size="small" style="width:140px" clearable @change="fetchAll"/>
    <el-button type="primary" size="small" style="margin-left:8px" @click="showWrite=true">+ 写入</el-button>
  </div>
</el-header>
<el-main style="padding:24px;background:#0d1117">
  <el-row :gutter="16">
    <el-col :span="8">
      <el-card style="background:#161b22;border-color:#30363d;margin-bottom:16px">
        <template #header><span style="color:#e6edf3">📊 总计</span></template>
        <div style="font-size:36px;color:#409eff;text-align:center">{{ stats.total_memories || '-' }}</div>
      </el-card>
    </el-col>
    <el-col :span="8">
      <el-card style="background:#161b22;border-color:#30363d;margin-bottom:16px">
        <template #header><span style="color:#e6edf3">🧠 嵌入</span></template>
        <div style="font-size:20px;text-align:center;margin-top:8px;color:#e6edf3">{{ stats.embedding_enabled ? '✅ 已启用' : '⛔ 未配置' }}</div>
      </el-card>
    </el-col>
    <el-col :span="8">
      <el-card style="background:#161b22;border-color:#30363d;margin-bottom:16px">
        <template #header><span style="color:#e6edf3">🔍 搜索</span></template>
        <el-input v-model="query" placeholder="关键词…" size="small" @keyup.enter="doSearch" style="--el-input-bg-color:#0d1117;--el-input-text-color:#e6edf3;--el-input-border-color:#30363d"/>
        <div v-if="results.length" style="margin-top:8px">
          <div v-for="r in results" :key="r.id" style="padding:4px 0;border-bottom:1px solid #30363d;font-size:13px;color:#e6edf3">
            <div>{{ r.content.slice(0,100) }}</div>
            <div style="display:flex;gap:4px;margin-top:2px">
              <el-tag size="small" :type="r.score>0.5?'primary':'info'">{{ (r.score||0).toFixed(3) }}</el-tag>
              <el-tag v-if="r.memory_type==='mental_model'" size="small" type="warning">🧠</el-tag>
            </div>
          </div>
        </div>
      </el-card>
    </el-col>
  </el-row>
  <el-card style="background:#161b22;border-color:#30363d;margin-top:16px">
    <template #header><span style="color:#e6edf3">📋 最近记忆</span></template>
    <el-table :data="recent" stripe max-height="500" style="width:100%">
      <el-table-column prop="id" label="ID" width="60"/>
      <el-table-column prop="content" label="内容" min-width="300" show-overflow-tooltip/>
      <el-table-column prop="memory_type" label="类型" width="100">
        <template #default="{row}"><el-tag :type="row.memory_type==='mental_model'?'warning':'primary'" size="small">{{ row.memory_type||'exp' }}</el-tag></template>
      </el-table-column>
      <el-table-column prop="hit_count" label="🔥" width="50" align="center"/>
      <el-table-column prop="created_at" label="时间" width="170"/>
    </el-table>
  </el-card>
</el-main>
<el-dialog v-model="showWrite" title="写入" width="400px">
  <el-input type="textarea" v-model="writeContent" :rows="3" placeholder="记忆内容…"/>
  <div style="margin-top:8px"><el-input v-model="writeAction" placeholder="动作(可选)" size="small"/></div>
  <template #footer>
    <el-button @click="showWrite=false">取消</el-button>
    <el-button type="primary" @click="doWrite">写入</el-button>
  </template>
</el-dialog>
</div>
<script src="/static/vue.global.prod.js"><\/script>
<script src="/static/element-plus.umd.js"><\/script>
<script src="/static/icons-vue.umd.js"><\/script>
<script>
const { createApp, ref, onMounted } = Vue;
const app = createApp({ setup() {
  const API = '/api/v1';
  const namespace = ref('default');
  const query = ref(''); const results = ref([]);
  const stats = ref({}); const recent = ref([]);
  const showWrite = ref(false); const writeContent = ref(''); const writeAction = ref('');
  async function fetchAll() {
    try {
      const s = await fetch(API+'/stats?namespace='+namespace.value); stats.value = await s.json();
      const r = await fetch(API+'/memories?namespace='+namespace.value+'&limit=20'); const d = await r.json(); recent.value = d.results||[];
    } catch(e) { console.error(e); }
  }
  async function doSearch() {
    try {
      const r = await fetch(API+'/search', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:query.value,limit:8})});
      const d = await r.json(); results.value = d.results||[];
    } catch(e) { console.error(e); }
  }
  async function doWrite() {
    try {
      await fetch(API+'/memories', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:writeContent.value,namespace:namespace.value,mem_action:writeAction.value||undefined})});
      showWrite.value=false; writeContent.value=''; writeAction.value=''; fetchAll();
    } catch(e) { console.error(e); }
  }
  onMounted(fetchAll);
  return {namespace,stats,recent,query,results,showWrite,writeContent,writeAction,fetchAll,doSearch,doWrite};
}});
for(const [k,c] of Object.entries(ElementPlusIconsVue)) app.component(k,c);
app.use(ElementPlus); app.mount('#app');
<\/script>
</body>
</html>"""


def get_store(db_path: str, embedding_dim: int = 1024) -> HybridMemoryStore:
    store = HybridMemoryStore(db_path=db_path, embedding_dim=embedding_dim)
    store.initialize()
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    logger.info("HMEM Server started: db_root=%s embed=%s", settings.db_root, bool(settings.embedding_base_url))
    yield


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(
        title="HMEM · 混合记忆系统",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
    )

    if settings.api_key:
        app.add_middleware(AuthMiddleware, api_key=settings.api_key)

    app.include_router(memories.router, prefix="/api/v1")
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(stats.router, prefix="/api/v1")
    app.include_router(graph.router, prefix="/api/v1")
    app.include_router(reflect.router, prefix="/api/v1")
    app.include_router(mental_models.router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.2.0"}

    webui_dist = os.path.join(os.path.dirname(__file__), "webui", "dist")
    webui_static = os.path.join(os.path.dirname(__file__), "webui", "static")
    if os.path.isdir(webui_dist):
        app.mount("/", StaticFiles(directory=webui_dist, html=True), name="webui")
    else:
        if os.path.isdir(webui_static):
            app.mount("/static", StaticFiles(directory=webui_static), name="static")
        @app.get("/")
        async def root():
            return HTMLResponse(content=SPA_HTML)

    return app


app = create_app()