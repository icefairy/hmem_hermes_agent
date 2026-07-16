<template>
  <div>
    <el-card>
      <template #header>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span>🧠 心智模型</span>
          <el-button type="warning" size="small" :icon="Lightning" @click="triggerReflect" :loading="reflecting">
            手动反思
          </el-button>
        </div>
      </template>
      <el-table :data="mentalModels" stripe style="width:100%" v-loading="loading" max-height="480">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="content" label="模式描述" min-width="350" show-overflow-tooltip />
        <el-table-column prop="hit_count" label="命中" width="60" align="center" />
        <el-table-column prop="created_at" label="创建时间" width="170" />
        <el-table-column label="操作" width="80" align="center">
          <template #default="{ row }">
            <el-button type="primary" link size="small" @click="viewDetails(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!loading && mentalModels.length === 0" description="尚无心智模型，请先写入记忆并触发反思" />
    </el-card>

    <!-- Reflect Result Dialog -->
    <el-dialog v-model="showResult" title="反思结果" width="500px">
      <div v-if="reflectResult">
        <el-alert
          :title="`生成 ${reflectResult.reflect_count || 0} 个心智模型`"
          :type="reflectResult.reflect_count > 0 ? 'success' : 'info'"
          show-icon
          style="margin-bottom:16px"
        />
        <div v-for="m in (reflectResult.models || [])" :key="m.model_id" style="margin-bottom:12px;padding:12px;background:#1c2128;border-radius:6px">
          <div style="font-size:13px;margin-bottom:4px">{{ m.pattern }}</div>
          <div style="display:flex;gap:8px">
            <el-tag size="small" type="warning">ID: {{ m.model_id }}</el-tag>
            <el-tag size="small" :type="m.confidence > 0.7 ? 'success' : 'info'">
              置信度: {{ (m.confidence || 0).toFixed(2) }}
            </el-tag>
            <el-tag size="small" v-if="m.supporting_count">支撑: {{ m.supporting_count }} 条</el-tag>
          </div>
        </div>
      </div>
    </el-dialog>

    <!-- Details Dialog -->
    <el-dialog v-model="showDetails" title="心智模型详情" width="650px">
      <template v-if="detailModel">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="ID">{{ detailModel.id }}</el-descriptions-item>
          <el-descriptions-item label="模式">{{ detailModel.content }}</el-descriptions-item>
          <el-descriptions-item label="命中次数">{{ detailModel.hit_count }}</el-descriptions-item>
          <el-descriptions-item label="创建时间">{{ detailModel.created_at }}</el-descriptions-item>
        </el-descriptions>
        <div v-if="detailChildren.length > 0" style="margin-top:16px">
          <h4 style="margin-bottom:8px">支撑经验 ({{ detailChildren.length }})</h4>
          <el-table :data="detailChildren" stripe max-height="300">
            <el-table-column prop="id" label="ID" width="60" />
            <el-table-column prop="content" label="内容" min-width="300" show-overflow-tooltip />
            <el-table-column prop="created_at" label="时间" width="170" />
          </el-table>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Lightning } from '@element-plus/icons-vue'

const props = defineProps<{ namespace: string }>()
const API_BASE = '/api/v1'

const mentalModels = ref<any[]>([])
const loading = ref(false)
const reflecting = ref(false)
const showResult = ref(false)
const reflectResult = ref<any>(null)
const showDetails = ref(false)
const detailModel = ref<any>(null)
const detailChildren = ref<any[]>([])

async function loadMentalModels() {
  loading.value = true
  try {
    const r = await fetch(`${API_BASE}/mental-models?namespace=${props.namespace}&limit=50`)
    const data = await r.json()
    mentalModels.value = data.results || []
  } catch { /* ignore */ }
  loading.value = false
}

async function triggerReflect() {
  reflecting.value = true
  try {
    const r = await fetch(`${API_BASE}/reflect?namespace=${props.namespace}`, { method: 'POST' })
    const data = await r.json()
    reflectResult.value = data
    showResult.value = true
    await loadMentalModels()
  } catch {
    ElMessage.error('反思触发失败')
  }
  reflecting.value = false
}

async function viewDetails(row: any) {
  detailModel.value = row
  try {
    const r = await fetch(`${API_BASE}/mental-models/${row.id}?namespace=${props.namespace}`)
    const data = await r.json()
    detailChildren.value = data.supporting_experiences || []
  } catch {
    detailChildren.value = []
  }
  showDetails.value = true
}

watch(() => props.namespace, loadMentalModels)
onMounted(loadMentalModels)
</script>