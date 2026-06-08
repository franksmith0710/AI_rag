<template>
  <div class="document-container">
    <div class="document-header">
      <h2>知识库管理</h2>
      <div class="header-actions">
        <span v-if="userStore.user" class="user-info">
          {{ userStore.user.username }} | {{ userStore.user.role === 'admin' ? 'Admin' : 'User' }}
        </span>
        <el-button @click="goToChat">
          <el-icon><ChatDotRound /></el-icon>
          返回对话
        </el-button>
        <el-upload
          :show-file-list="false"
          :http-request="handleUpload"
          multiple
          accept=".pdf,.docx,.doc,.txt,.md,.jpg,.jpeg,.png,.bmp,.tiff"
        >
          <el-button type="primary" :loading="uploading">
            <el-icon><Upload /></el-icon>
            上传文档
          </el-button>
        </el-upload>
        <el-button
          type="success"
          :disabled="selectedIds.length === 0"
          :loading="batchProcessing"
          @click="batchProcess"
        >
          批量处理 ({{ selectedIds.length }})
        </el-button>
      </div>
    </div>

    <el-tabs v-model="activeTab" class="doc-tabs" @tab-change="handleTabChange">
      <el-tab-pane label="我的文档" name="my" />
      <el-tab-pane label="全局共享" name="global" />
    </el-tabs>

    <el-table :data="documents" style="width: 100%" v-loading="loading" @selection-change="handleSelectionChange">
      <el-table-column type="selection" width="50" :selectable="(row) => row.status !== 'completed'" />
      <el-table-column prop="title" label="文档标题" min-width="200" />
      <el-table-column prop="file_name" label="文件名" min-width="150" />
      <el-table-column prop="file_type" label="类型" width="80" />
      <el-table-column prop="chunk_count" label="Chunk数" width="80" />
      <el-table-column prop="status" label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="row.status === 'completed' ? 'success' : 'warning'">
            {{ row.status === 'completed' ? '已处理' : '待处理' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="上传时间" width="180">
        <template #default="{ row }">
          {{ formatTime(row.created_at) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="280" fixed="right">
        <template #default="{ row }">
          <el-button
            size="small"
            type="info"
            @click="viewChunks(row)"
            :disabled="row.status !== 'completed'"
          >
            查看Chunks
          </el-button>
          <el-button
            v-if="row.status !== 'completed'"
            size="small"
            @click="processDocument(row.id)"
            :loading="processingId === row.id"
          >
            处理
          </el-button>
          <el-button
            size="small"
            type="danger"
            @click="deleteDocument(row)"
            :disabled="row.tenant_id === 0 && !isAdmin"
          >
            删除
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination
      v-if="total > pageSize"
      class="pagination"
      layout="prev, pager, next"
      :total="total"
      :page-size="pageSize"
      :current-page="currentPage"
      @current-change="handlePageChange"
    />

    <el-dialog
      v-model="chunksDialogVisible"
      :title="`查看Chunks: ${currentDocTitle}`"
      width="80%"
      destroy-on-close
    >
      <div class="chunks-container">
        <div class="chunks-list">
          <div
            v-for="chunk in chunks"
            :key="chunk.chunk_index"
            class="chunk-item"
            :class="{ active: selectedChunkIndex === chunk.chunk_index }"
            @click="selectedChunkIndex = chunk.chunk_index"
          >
            <div class="chunk-index">#{{ chunk.chunk_index + 1 }}</div>
            <div class="chunk-preview">{{ chunk.text.length > 50 ? chunk.text.slice(0, 50) + '...' : chunk.text }}</div>
          </div>
        </div>
        <div class="chunk-content">
          <div v-if="chunks.length > 0" class="chunk-text">
            <pre>{{ chunks.find(c => c.chunk_index === selectedChunkIndex)?.text || '' }}</pre>
          </div>
          <div v-else class="no-chunks">暂无Chunks数据</div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '../stores/user'
import api from '../api'
import { ElMessage, ElMessageBox } from 'element-plus'

const router = useRouter()
const userStore = useUserStore()

const documents = ref([])
const loading = ref(false)
const processingId = ref(null)
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)
const activeTab = ref('my')
const chunksDialogVisible = ref(false)
const currentDocTitle = ref('')
const chunks = ref([])
const selectedChunkIndex = ref(0)

const uploading = ref(false)
const selectedIds = ref([])
const batchProcessing = ref(false)

const isAdmin = computed(() => userStore.user?.role === 'admin')

const handleSelectionChange = (rows) => {
  selectedIds.value = rows.map(r => r.id)
}

const fetchDocuments = async () => {
  loading.value = true
  try {
    const params = {
      skip: (currentPage.value - 1) * pageSize.value,
      limit: pageSize.value
    }
    // 服务端过滤：is_global=true 仅全局，is_global=false 仅个人，不传=全部
    if (activeTab.value === 'global') {
      params.is_global = 'true'
    } else {
      params.is_global = 'false'
    }
    const response = await api.get('/api/documents', { params })
    documents.value = response.data.items
    total.value = response.data.total
  } catch (error) {
    ElMessage.error('获取文档列表失败')
  } finally {
    loading.value = false
  }
}

const handleTabChange = () => {
  currentPage.value = 1
  fetchDocuments()
}

const handleUpload = async ({ file, onSuccess, onError }) => {
  uploading.value = true
  try {
    const formData = new FormData()
    formData.append('files', file)
    formData.append('is_global', activeTab.value === 'global' ? 'true' : 'false')

    const response = await api.post('/api/documents/upload-batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000
    })

    if (response.data.code === 200) {
      const data = response.data.data
      if (data.success > 0) {
        ElMessage.success(`上传成功: ${data.success} 个文件`)
      } else {
        for (const r of data.results) {
          if (!r.success) ElMessage.error(`${r.file_name}: ${r.error}`)
        }
      }
    } else {
      ElMessage.error(response.data.message || '上传失败')
    }

    onSuccess()
    await fetchDocuments()
  } catch (error) {
    onError(error)
    ElMessage.error(error.response?.data?.message || '上传失败')
  } finally {
    uploading.value = false
  }
}

const batchProcess = async () => {
  if (selectedIds.value.length === 0) return
  batchProcessing.value = true
  try {
    const response = await api.post('/api/documents/process-batch', {
      document_ids: selectedIds.value
    }, { timeout: 300000 })
    if (response.data.code === 200) {
      const data = response.data.data
      if (data.failed > 0) {
        ElMessage.warning(`批量处理完成: ${data.success} 成功, ${data.failed} 失败`)
      } else {
        ElMessage.success(`批量处理完成: ${data.success} 个文档`)
      }
    } else {
      ElMessage.error(response.data.message || '批量处理失败')
    }
    await fetchDocuments()
  } catch (error) {
    ElMessage.error(error.response?.data?.message || '批量处理失败')
  } finally {
    batchProcessing.value = false
  }
}

const processDocument = async (docId) => {
  processingId.value = docId
  try {
    await api.post(`/api/documents/process/${docId}`, null, { timeout: 120000 })
    ElMessage.success('处理成功')
    await fetchDocuments()
  } catch (error) {
    ElMessage.error(error.response?.data?.message || '处理失败')
  } finally {
    processingId.value = null
  }
}

const deleteDocument = async (doc) => {
  // 权限校验
  if (doc.tenant_id === 0 && !isAdmin.value) {
    ElMessage.warning('只有管理员可以删除全局共享文档')
    return
  }

  try {
    await ElMessageBox.confirm('确定要删除这个文档吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })

    await api.delete(`/api/documents/${doc.id}`)
    ElMessage.success('删除成功')
    await fetchDocuments()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error(error.response?.data?.message || '删除失败')
    }
  }
}

const handlePageChange = (page) => {
  currentPage.value = page
  fetchDocuments()
}

const formatTime = (time) => {
  if (!time) return ''
  return new Date(time).toLocaleString('zh-CN')
}

const goToChat = () => {
  router.push('/chat')
}

const viewChunks = async (doc) => {
  currentDocTitle.value = doc.title
  chunksDialogVisible.value = true
  chunks.value = []
  selectedChunkIndex.value = 0
  try {
    const response = await api.get(`/api/documents/${doc.id}/chunks`)
    chunks.value = response.data.items
    if (chunks.value.length > 0) {
      selectedChunkIndex.value = chunks.value[0].chunk_index
    }
  } catch (error) {
    ElMessage.error('获取Chunks失败')
  }
}

onMounted(async () => {
  await userStore.fetchCurrentUser()
  await fetchDocuments()
})
</script>

<style scoped>
.document-container {
  padding: 20px;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.document-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.document-header h2 {
  margin: 0;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.header-actions .user-info {
  margin-right: auto;
  padding: 6px 12px;
  background: #f0f0f0;
  border-radius: 4px;
  font-size: 14px;
  color: #333;
}

.doc-tabs {
  margin-bottom: 20px;
}

.pagination {
  margin-top: 20px;
  justify-content: center;
}

.chunks-container {
  display: flex;
  height: 500px;
  gap: 20px;
}

.chunks-list {
  width: 300px;
  overflow-y: auto;
  border-right: 1px solid #eee;
  padding-right: 10px;
}

.chunk-item {
  padding: 10px;
  margin-bottom: 8px;
  border-radius: 4px;
  cursor: pointer;
  background: #f5f7fa;
  transition: all 0.2s;
}

.chunk-item:hover {
  background: #e6f0ff;
}

.chunk-item.active {
  background: #409eff;
  color: white;
}

.chunk-item.active .chunk-preview {
  color: rgba(255, 255, 255, 0.8);
}

.chunk-index {
  font-weight: bold;
  margin-bottom: 4px;
}

.chunk-preview {
  font-size: 12px;
  color: #666;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chunk-content {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
}

.chunk-text pre {
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 14px;
  line-height: 1.6;
  margin: 0;
}

.no-chunks {
  text-align: center;
  color: #999;
  margin-top: 50px;
}
</style>