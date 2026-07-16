<template>
  <div>
    <el-row :gutter="16">
      <el-col :span="8">
        <el-card>
          <template #header><span>📊 记忆总数</span></template>
          <div style="font-size:36px;color:#409eff;text-align:center">{{ stats.total_memories ?? '-' }}</div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header><span>🧠 嵌入引擎</span></template>
          <div style="font-size:24px;text-align:center;color:var(--el-color-primary);margin-top:8px">
            {{ stats.embedding_enabled ? '✅ 已启用' : '⛔ 未配置' }}
          </div>
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card>
          <template #header><span>🏷️ 记忆类型</span></template>
          <div v-if="stats.by_type" style="font-size:14px">
            <div v-for="(cnt, typ) in stats.by_type" :key="typ" style="display:flex;justify-content:space-between;padding:4px 0">
              <span>{{ typ }}</span>
              <el-tag size="small">{{ cnt }}</el-tag>
            </div>
          </div>
          <div v-else style="text-align:center;color:#8b949e">暂无数据</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card style="margin-top:16px">
      <template #header>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span>📋 最近记忆</span>
          <el-button size="small" @click="loadRecent">刷新</el-button>
        </div>
      </template>
      <el-table :data="recent" stripe style="width:100%" v-loading="loadingRecent" max-height="480">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="content" label="内容" min-width="300" show-overflow-tooltip />
        <el-table-column prop="memory_type" label="类型" width="100">
          <template #default="{ row }">
            <el-tag :type="row.memory_type === 'mental_model' ? 'warning' : 'primary'" size="small">
              {{ row.memory_type || 'experience' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="hit_count" label="命中" width="60" align="center" />
        <el-table-column prop="created_at" label="创建时间" width="180" />
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'

const props = defineProps<{ namespace: string }>()
const API_BASE = '/api/v1'

const stats = ref<any>({})
const recent = ref<any[]>([])
const loadingRecent = ref(false)

async function fetchStats() {
  try {
    const r = await fetch(`${API_BASE}/stats?namespace=${props.namespace}`)
    stats.value = await r.json()
  } catch { /* ignore */ }
}

async function loadRecent() {
  loadingRecent.value = true
  try {
    const r = await fetch(`${API_BASE}/memories?namespace=${props.namespace}&limit=20`)
    const data = await r.json()
    recent.value = data.results || []
  } catch { /* ignore */ }
  loadingRecent.value = false
}

watch(() => props.namespace, () => { fetchStats(); loadRecent() })
onMounted(() => { fetchStats(); loadRecent() })
</script>