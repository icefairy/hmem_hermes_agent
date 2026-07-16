<template>
  <div id="hmem-app">
    <el-container>
      <el-header>
        <h1>🧠 HMEM · 混合记忆管理系统</h1>
        <p style="color:#aaa;font-size:14px">SQLite + sqlite-vec + jieba + bge-m3</p>
      </el-header>
      <el-main>
        <el-row :gutter="20">
          <el-col :span="6">
            <el-card>
              <template #header><span>📊 概览</span></template>
              <el-skeleton :loading="loading" :count="3">
                <div v-for="s in stats" :key="s.label" style="margin:8px 0">
                  <strong>{{ s.label }}</strong><br/>
                  <span style="font-size:24px;color:#409eff">{{ s.value }}</span>
                </div>
              </el-skeleton>
            </el-card>
          </el-col>
          <el-col :span="18">
            <el-card>
              <template #header>
                <div style="display:flex;justify-content:space-between;align-items:center">
                  <span>🔍 记忆检索</span>
                  <el-button type="primary" size="small" @click="showWriteDialog = true">+ 写入</el-button>
                </div>
              </template>
              <el-input
                v-model="searchQuery"
                placeholder="输入关键词搜索记忆…"
                clearable
                @keyup.enter="doSearch"
              >
                <template #append>
                  <el-button @click="doSearch">搜索</el-button>
                </template>
              </el-input>
              <div v-if="searchResults.length > 0" style="margin-top:16px">
                <el-timeline>
                  <el-timeline-item
                    v-for="item in searchResults"
                    :key="item.id"
                    :timestamp="item.created_at"
                    :color="item.score > 0.3 ? '#409eff' : '#ddd'"
                  >
                    <div style="display:flex;justify-content:space-between">
                      <span>{{ item.content }}</span>
                      <el-tag :type="item.score > 0.3 ? 'primary' : 'info'" size="small">
                        {{ (item.score || 0).toFixed(3) }}
                      </el-tag>
                    </div>
                    <div style="font-size:12px;color:#999;margin-top:4px">
                      空间: {{ item.agent_space }} · ID: {{ item.id }}
                    </div>
                  </el-timeline-item>
                </el-timeline>
              </div>
              <el-empty v-else-if="searched" description="未找到匹配的记忆" />
            </el-card>
          </el-col>
        </el-row>
      </el-main>
    </el-container>

    <!-- Write Dialog -->
    <el-dialog v-model="showWriteDialog" title="写入记忆" width="500px">
      <el-form>
        <el-form-item label="内容">
          <el-input type="textarea" v-model="writeContent" :rows="4" />
        </el-form-item>
        <el-form-item label="空间">
          <el-input v-model="writeSpace" placeholder="default" />
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
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'

const API_BASE = '/api'

const loading = ref(true)
const stats = ref([{ label: '总计', value: '-' }, { label: '当前空间', value: '-' }, { label: '嵌入引擎', value: '-' }])

const searchQuery = ref('')
const searchResults = ref<any[]>([])
const searched = ref(false)

const showWriteDialog = ref(false)
const writeContent = ref('')
const writeSpace = ref('default')
const writing = ref(false)

async function fetchStats() {
  try {
    const r = await fetch(`${API_BASE}/stats`)
    const data = await r.json()
    stats.value = [
      { label: '记忆总数', value: data.total_memories },
      { label: '当前空间', value: `${data.current_space} (${data.current_space_count})` },
      { label: '嵌入引擎', value: data.embedding_enabled ? '✅ 已启用' : '⛔ 未启用' },
    ]
  } catch { /* ignore */ }
  loading.value = false
}

async function doSearch() {
  if (!searchQuery.value.trim()) return
  searched.value = true
  searchResults.value = []
  try {
    const r = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: searchQuery.value, limit: 20 }),
    })
    const data = await r.json()
    searchResults.value = data.results || []
  } catch {
    ElMessage.error('搜索请求失败')
  }
}

async function doWrite() {
  if (!writeContent.value.trim()) return
  writing.value = true
  try {
    const r = await fetch(`${API_BASE}/write`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: writeContent.value, agent_space: writeSpace.value }),
    })
    const data = await r.json()
    ElMessage.success(`写入成功 (ID: ${data.memory_id})`)
    showWriteDialog.value = false
    writeContent.value = ''
    fetchStats()
  } catch {
    ElMessage.error('写入失败')
  }
  writing.value = false
}

onMounted(fetchStats)
</script>

<style>
body { margin: 0; background: #0d1117; color: #e6edf3; font-family: system-ui, sans-serif; }
#hmem-app { min-height: 100vh; }
.el-header { border-bottom: 1px solid #30363d; background: #161b22; }
.el-main { padding: 24px; }
.el-card { background: #161b22; border-color: #30363d; color: #e6edf3; margin-bottom: 16px; }
.el-card >>> .el-card__header { border-bottom-color: #30363d; }
.el-input, .el-textarea { --el-input-bg-color: #0d1117; --el-input-border-color: #30363d; --el-input-text-color: #e6edf3; --el-input-hover-border-color: #409eff; }
.el-skeleton >>> .el-skeleton__item { background: #21262d; }
.el-timeline >>> .el-timeline-item__content { color: #e6edf3; }
.el-timeline >>> .el-timeline-item__timestamp { color: #8b949e; }
</style>