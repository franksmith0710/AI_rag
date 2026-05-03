<template>
  <div class="document-container">
    <div class="document-header">
      <h2>知识库管理</h2>
      <div class="header-actions">
        <el-button @click="goToChat">
          <el-icon><ChatDotRound /></el-icon>
          返回对话
        </el-button>
        <el-upload
          :action="uploadUrl"
          :headers="uploadHeaders"
          :on-success="handleUploadSuccess"
          :on-error="handleUploadError"
          :show-file-list="false"
          accept=".pdf,.docx,.doc,.txt"
        >
          <el-button type="primary">
            <el-icon><Upload /></el-icon>
            上传文档
          </el-button>
        </el-upload>
      </div>
    </div>

    <el-table :data="documents" style="width: 100%" v-loading="loading">
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
      <el-table-column label="操作" width="200" fixed="right">
        <template #default="{ row }">
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
            @click="deleteDocument(row.id)"
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
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import api from '../api'
import { ElMessage, ElMessageBox } from 'element-plus'

const router = useRouter()

const documents = ref([])
const loading = ref(false)
const processingId = ref(null)
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)

const uploadUrl = computed(() => '/api/documents/upload')
const uploadHeaders = computed(() => ({
  Authorization: `Bearer ${localStorage.getItem('token')}`
}))

const fetchDocuments = async () => {
  loading.value = true
  try {
    const response = await api.get('/api/documents', {
      params: {
        skip: (currentPage.value - 1) * pageSize.value,
        limit: pageSize.value
      }
    })
    documents.value = response.data.items
    total.value = response.data.total
  } catch (error) {
    ElMessage.error('获取文档列表失败')
  } finally {
    loading.value = false
  }
}

const handleUploadSuccess = async (response) => {
  ElMessage.success('上传成功')
  await fetchDocuments()

  if (response.status === 'completed') {
    ElMessage.success('文档已自动处理完成')
  } else {
    ElMessage.info('文档已上传，请点击"处理"按钮进行向量化')
  }
}

const handleUploadError = (error) => {
  ElMessage.error(error.response?.data?.message || '上传失败')
}

const processDocument = async (docId) => {
  processingId.value = docId
  try {
    await api.post(`/api/documents/process/${docId}`)
    ElMessage.success('处理成功')
    await fetchDocuments()
  } catch (error) {
    ElMessage.error(error.response?.data?.message || '处理失败')
  } finally {
    processingId.value = null
  }
}

const deleteDocument = async (docId) => {
  try {
    await ElMessageBox.confirm('确定要删除这个文档吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })

    await api.delete(`/api/documents/${docId}`)
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

onMounted(() => {
  fetchDocuments()
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
  gap: 10px;
}

.pagination {
  margin-top: 20px;
  justify-content: center;
}
</style>