<template>
  <div id="hmem-app">
    <el-container>
      <el-aside width="60px" style="background:#161b22;border-right:1px solid #30363d;">
        <el-menu
          :default-active="currentView"
          style="border:none;background:transparent"
          @select="(k) => currentView = k"
        >
          <el-menu-item index="dashboard">
            <el-icon><DataBoard /></el-icon>
            <template #title>概览</template>
          </el-menu-item>
          <el-menu-item index="search">
            <el-icon><Search /></el-icon>
            <template #title>搜索</template>
          </el-menu-item>
          <el-menu-item index="graph">
            <el-icon><Share /></el-icon>
            <template #title>图谱</template>
          </el-menu-item>
          <el-menu-item index="reflect">
            <el-icon><Lightning /></el-icon>
            <template #title>反思</template>
          </el-menu-item>
        </el-menu>
      </el-aside>
      <el-container>
        <el-header style="display:flex;align-items:center;justify-content:space-between;padding:0 24px">
          <div style="display:flex;align-items:center;gap:12px">
            <h2 style="margin:0;font-size:18px">🧠 HMEM</h2>
            <el-tag size="small" type="info">{{ activeNamespace || 'default' }}</el-tag>
          </div>
          <div style="display:flex;align-items:center;gap:8px">
            <el-input
              v-model="activeNamespace"
              placeholder="namespace"
              size="small"
              style="width:160px"
              clearable
            />
            <el-button size="small" :icon="Plus" @click="showWriteDialog = true">写入</el-button>
          </div>
        </el-header>
        <el-main>
          <Dashboard v-if="currentView === 'dashboard'" :namespace="activeNamespace" />
          <SearchView v-else-if="currentView === 'search'" :namespace="activeNamespace" />
          <GraphView v-else-if="currentView === 'graph'" :namespace="activeNamespace" />
          <ReflectView v-else-if="currentView === 'reflect'" :namespace="activeNamespace" />
        </el-main>
      </el-container>
    </el-container>

    <!-- Write Dialog -->
    <el-dialog v-model="showWriteDialog" title="写入记忆" width="500px">
      <el-form label-width="80px">
        <el-form-item label="内容">
          <el-input type="textarea" v-model="writeContent" :rows="4" />
        </el-form-item>
        <el-form-item label="动作">
          <el-input v-model="writeAction" placeholder="code_generation / qa / debug …" />
        </el-form-item>
        <el-form-item label="空间">
          <el-input v-model="writeNamespace" placeholder="default" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showWriteDialog = false">取消</el-button>
        <el-button type="primary" @click="doWrite" :loading="writing">写入</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import Dashboard from './views/Dashboard.vue'
import SearchView from './views/SearchView.vue'
import GraphView from './views/GraphView.vue'
import ReflectView from './views/ReflectView.vue'

const API_BASE = '/api/v1'
const currentView = ref('dashboard')
const activeNamespace = ref('default')

const showWriteDialog = ref(false)
const writeContent = ref('')
const writeAction = ref('')
const writeNamespace = ref('default')
const writing = ref(false)

async function doWrite() {
  if (!writeContent.value.trim()) return
  writing.value = true
  try {
    const r = await fetch(`${API_BASE}/memories`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: writeContent.value,
        namespace: writeNamespace.value || 'default',
        mem_action: writeAction.value || undefined,
      }),
    })
    const data = await r.json()
    ElMessage.success(`写入成功 (ID: ${data.memory_id})`)
    showWriteDialog.value = false
    writeContent.value = ''
    writeAction.value = ''
  } catch {
    ElMessage.error('写入失败')
  }
  writing.value = false
}
</script>

<style>
body { margin: 0; background: #0d1117; color: #e6edf3; font-family: system-ui, sans-serif; }
#hmem-app { min-height: 100vh; }
.el-header { border-bottom: 1px solid #30363d; background: #161b22; height: 56px; }
.el-main { padding: 24px; background: #0d1117; }
.el-card { background: #161b22; border-color: #30363d; color: #e6edf3; margin-bottom: 16px; }
.el-card :deep(.el-card__header) { border-bottom-color: #30363d; }
.el-table { --el-table-bg-color: #161b22; --el-table-tr-bg-color: #161b22; --el-table-header-bg-color: #21262d; --el-table-border-color: #30363d; --el-table-text-color: #e6edf3; --el-table-header-text-color: #e6edf3; --el-table-row-hover-bg-color: #1c2128; }
.el-input, .el-textarea { --el-input-bg-color: #0d1117; --el-input-border-color: #30363d; --el-input-text-color: #e6edf3; --el-input-hover-border-color: #409eff; --el-focus-border-color: #409eff; }
.el-menu { --el-menu-bg-color: transparent; --el-menu-text-color: #8b949e; --el-menu-active-color: #409eff; --el-menu-hover-bg-color: #1c2128; --el-menu-item-height: 48px; }
.el-menu-item.is-active { background: #1c2128 !important; }
.el-dialog { --el-dialog-bg-color: #161b22; --el-dialog-title-text-color: #e6edf3; --el-dialog-content-text-color: #e6edf3; --el-border-color: #30363d; }
.el-tag { --el-tag-bg-color: #1c2128; --el-tag-text-color: #8b949e; --el-tag-border-color: #30363d; }
.el-card__body { padding: 16px; }
</style>