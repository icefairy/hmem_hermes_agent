<template>
  <div>
    <el-card>
      <template #header><span>🔍 混合检索</span></template>
      <el-input
        v-model="query"
        placeholder="输入关键词或自然语言查询…"
        clearable
        size="large"
        @keyup.enter="doSearch"
      >
        <template #append>
          <el-button type="primary" @click="doSearch" :loading="searching">搜索</el-button>
        </template>
      </el-input>
      <div style="margin-top:12px;display:flex;gap:8px;align-items:center">
        <el-checkbox v-model="useRerank" label="启用重排" size="small" />
        <el-select v-model="searchLimit" size="small" style="width:100px">
          <el-option v-for="n in [5,10,20,50]" :key="n" :label="`Top ${n}`" :value="n" />
        </el-select>
      </div>
    </el-card>

    <el-card v-if="results.length > 0" style="margin-top:16px">
      <template #header>
        <span>📄 结果 ({{ results.length }})</span>
      </template>
      <el-table :data="results" stripe style="width:100%" max-height="600">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="content" label="内容" min-width="300" show-overflow-tooltip />
        <el-table-column prop="score" label="得分" width="80" align="center">
          <template #default="{ row }">
            <el-tag :type="row.score > 0.5 ? 'primary' : 'info'" size="small">
              {{ (row.score || 0).toFixed(3) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="来源" width="120" align="center">
          <template #default="{ row }">
            <div style="display:flex;gap:4px;justify-content:center">
              <el-tag v-if="row.fts_rank != null" size="small" type="success">K</el-tag>
              <el-tag v-if="row.vec_distance != null" size="small" type="warning">V</el-tag>
              <el-tag v-if="row.score != null && row.fts_rank == null && row.vec_distance == null" size="small" type="danger">R</el-tag>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="memory_type" label="类型" width="90">
          <template #default="{ row }">
            <el-tag :type="row.memory_type === 'mental_model' ? 'warning' : 'primary'" size="small">
              {{ row.memory_type || 'experience' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="时间" width="170" />
      </el-table>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'

const props = defineProps<{ namespace: string }>()
const API_BASE = '/api/v1'

const query = ref('')
const results = ref<any[]>([])
const searching = ref(false)
const useRerank = ref(true)
const searchLimit = ref(10)

async function doSearch() {
  if (!query.value.trim()) return
  searching.value = true
  try {
    const r = await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query.value,
        limit: searchLimit.value,
        use_rerank: useRerank.value,
      }),
    })
    const data = await r.json()
    results.value = data.results || []
  } catch {
    ElMessage.error('搜索失败')
  }
  searching.value = false
}
</script>