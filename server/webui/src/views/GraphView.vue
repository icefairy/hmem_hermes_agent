<template>
  <div>
    <el-card>
      <template #header>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span>🕸️ 记忆关系图谱</span>
          <el-button size="small" :icon="Refresh" @click="loadGraph" :loading="loading">刷新</el-button>
        </div>
      </template>
      <div ref="chartRef" style="width:100%;height:600px;background:#0d1117;border-radius:8px" />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, nextTick } from 'vue'
import { Refresh } from '@element-plus/icons-vue'
import * as echarts from 'echarts'

const props = defineProps<{ namespace: string }>()
const API_BASE = '/api/v1'

const chartRef = ref<HTMLDivElement>()
const loading = ref(false)
let chart: echarts.ECharts | null = null

async function loadGraph() {
  loading.value = true
  try {
    const r = await fetch(`${API_BASE}/graph?namespace=${props.namespace}`)
    const data = await r.json()
    renderGraph(data)
  } catch { /* ignore */ }
  loading.value = false
}

function renderGraph(data: any) {
  if (!chartRef.value) return

  if (!chart) {
    chart = echarts.init(chartRef.value, undefined, { renderer: 'canvas' })
  }

  const nodes = (data.nodes || []).map((n: any) => ({
    id: String(n.id),
    name: n.label || n.id,
    symbolSize: Math.max(10, Math.min(40, 10 + (n.hit_count || 0) * 3)),
    category: n.type === 'mental_model' ? 1 : 0,
    itemStyle: n.type === 'mental_model'
      ? { color: '#e6a23c' }
      : { color: '#409eff' },
    label: { show: n.type === 'mental_model', fontSize: 11 },
  }))

  // 节点 ID 集合，用于过滤边
  const nodeIds = new Set(nodes.map((n: any) => n.id))
  const edges = (data.edges || [])
    .filter((e: any) => nodeIds.has(String(e.source)) && nodeIds.has(String(e.target)))
    .map((e: any) => ({
      source: String(e.source),
      target: String(e.target),
      value: e.weight || 1,
      lineStyle: { width: (e.weight || 1) * 2, opacity: 0.3, curveness: 0.2 },
    }))

  chart.setOption({
    backgroundColor: '#0d1117',
    tooltip: { trigger: 'item', formatter: '{b}' },
    legend: {
      data: ['记忆', '心智模型'],
      textStyle: { color: '#8b949e' },
      bottom: 0,
    },
    series: [{
      type: 'graph',
      layout: 'force',
      force: { repulsion: 300, edgeLength: 120, gravity: 0.1 },
      roam: true,
      draggable: true,
      data: nodes,
      edges: edges,
      categories: [
        { name: '记忆', itemStyle: { color: '#409eff' } },
        { name: '心智模型', itemStyle: { color: '#e6a23c' } },
      ],
      label: { show: false, position: 'right', color: '#8b949e', fontSize: 10 },
      edgeLabel: { show: false },
      lineStyle: { color: '#30363d', opacity: 0.2 },
    }],
  })
}

watch(() => props.namespace, () => nextTick(loadGraph))
onMounted(() => nextTick(loadGraph))
</script>