<template>
  <div class="chat-container">
    <div class="sidebar">
      <div class="sidebar-header">
        <el-button type="primary" @click="createSession" style="width: 100%">
          新建会话
        </el-button>
      </div>
      <el-scrollbar class="session-list">
        <div
          v-for="session in sessions"
          :key="session.id"
          :class="['session-item', { active: currentSessionId === session.id }]"
          @click="selectSession(session)"
        >
          <div class="session-info">
            <div class="session-title">{{ session.title }}</div>
            <div class="session-time">{{ formatTime(session.updated_at) }}</div>
          </div>
          <el-button
            class="delete-btn"
            type="danger"
            size="small"
            :icon="Delete"
            circle
            @click.stop="deleteSession(session.id)"
          />
        </div>
      </el-scrollbar>
      <div class="sidebar-footer">
        <el-button text @click="goToDocuments">
          <el-icon><Folder /></el-icon>
          知识库管理
        </el-button>
        <el-button text @click="handleLogout">
          <el-icon><SwitchButton /></el-icon>
          退出登录
        </el-button>
      </div>
    </div>

    <div class="chat-main">
      <div class="chat-header">
        <span>{{ currentSession?.title || '请选择会话' }}</span>
      </div>

      <el-scrollbar class="chat-messages" ref="messagesScroll">
        <div class="messages-wrapper">
          <div
            v-for="msg in messages"
            :key="msg.id"
            :class="['message', msg.role]"
          >
            <div class="message-content">
              <div class="message-text">{{ msg.content }}</div>
              <div v-if="msg.sources && msg.sources.length" class="message-sources">
                <div class="sources-title">参考文档：</div>
                <div v-for="(source, idx) in msg.sources" :key="idx" class="source-item">
                  {{ source.text?.substring(0, 100) }}...
                </div>
              </div>
            </div>
          </div>
          <div v-if="loading" class="message assistant">
            <div class="message-content">
              <div class="loading">正在思考...</div>
            </div>
          </div>
        </div>
      </el-scrollbar>

      <div class="chat-input">
        <el-input
          v-model="inputMessage"
          type="textarea"
          :rows="3"
          placeholder="请输入问题，按 Enter 发送..."
          @keydown.enter="sendMessage"
          :disabled="!currentSessionId || loading"
        />
        <el-button
          type="primary"
          :loading="loading"
          :disabled="!currentSessionId || !inputMessage.trim()"
          @click="sendMessage"
        >
          发送
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '../stores/user'
import api from '../api'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete } from '@element-plus/icons-vue'

const router = useRouter()
const userStore = useUserStore()

const sessions = ref([])
const currentSessionId = ref(null)
const currentSession = ref(null)
const messages = ref([])
const inputMessage = ref('')
const loading = ref(false)
const messagesScroll = ref(null)

const fetchSessions = async () => {
  try {
    const response = await api.get('/api/sessions')
    sessions.value = response.data.items
  } catch (error) {
    console.error('获取会话列表失败', error)
  }
}

const createSession = async () => {
  try {
    const response = await api.post('/api/sessions', {})
    sessions.value.unshift(response.data)
    selectSession(response.data)
  } catch (error) {
    ElMessage.error('创建会话失败')
  }
}

const deleteSession = async (sessionId) => {
  try {
    await ElMessageBox.confirm('确定要删除这个会话吗？', '提示', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })
    await api.delete(`/api/sessions/${sessionId}`)
    ElMessage.success('删除成功')
    sessions.value = sessions.value.filter(s => s.id !== sessionId)
    if (currentSessionId.value === sessionId) {
      currentSessionId.value = null
      currentSession.value = null
      messages.value = []
      if (sessions.value.length > 0) {
        selectSession(sessions.value[0])
      }
    }
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

const selectSession = async (session) => {
  currentSessionId.value = session.id
  currentSession.value = session
  await fetchMessages()
}

const fetchMessages = async () => {
  if (!currentSessionId.value) return
  try {
    const response = await api.get(`/api/chat/history/${currentSessionId.value}`)
    messages.value = response.data.data || []
    scrollToBottom()
  } catch (error) {
    console.error('获取消息失败', error)
  }
}

const sendMessage = async () => {
  if (!inputMessage.value.trim() || !currentSessionId.value || loading.value) return

  const userMessage = inputMessage.value.trim()
  inputMessage.value = ''
  loading.value = true

  try {
    const response = await api.post('/api/chat', {
      session_id: currentSessionId.value,
      message: userMessage
    })

    messages.value.push({ role: 'user', content: userMessage })
    messages.value.push({
      role: 'assistant',
      content: response.data.message,
      sources: response.data.sources
    })
    scrollToBottom()
  } catch (error) {
    ElMessage.error(error.response?.data?.message || '发送消息失败')
  } finally {
    loading.value = false
  }
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesScroll.value) {
      messagesScroll.value.setScrollTop(999999)
    }
  })
}

const formatTime = (time) => {
  if (!time) return ''
  const date = new Date(time)
  return `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`
}

const goToDocuments = () => {
  router.push('/documents')
}

const handleLogout = () => {
  userStore.logout()
  router.push('/login')
}

onMounted(async () => {
  await userStore.fetchCurrentUser()
  await fetchSessions()
  if (sessions.value.length > 0) {
    selectSession(sessions.value[0])
  }
})
</script>

<style scoped>
.chat-container {
  display: flex;
  height: 100vh;
}

.sidebar {
  width: 250px;
  background: #f5f7fa;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 15px;
  border-bottom: 1px solid #e4e7ed;
}

.session-list {
  flex: 1;
  overflow: hidden;
}

.session-item {
  padding: 12px 15px;
  cursor: pointer;
  border-bottom: 1px solid #e4e7ed;
  transition: background 0.3s;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.session-info {
  flex: 1;
  overflow: hidden;
}

.delete-btn {
  opacity: 0;
  transition: opacity 0.3s;
}

.session-item:hover .delete-btn {
  opacity: 1;
}

.session-item:hover {
  background: #ecf5ff;
}

.session-item.active {
  background: #409eff;
  color: white;
}

.session-item.active .session-time {
  color: rgba(255, 255, 255, 0.8);
}

.session-title {
  font-size: 14px;
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-time {
  font-size: 12px;
  color: #909399;
}

.sidebar-footer {
  padding: 10px;
  border-top: 1px solid #e4e7ed;
  display: flex;
  justify-content: space-between;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.chat-header {
  padding: 15px 20px;
  border-bottom: 1px solid #e4e7ed;
  font-size: 16px;
  font-weight: bold;
}

.chat-messages {
  flex: 1;
  padding: 20px;
  overflow: hidden;
}

.messages-wrapper {
  max-width: 800px;
  margin: 0 auto;
}

.message {
  display: flex;
  margin-bottom: 20px;
}

.message.user {
  justify-content: flex-end;
}

.message.assistant {
  justify-content: flex-start;
}

.message-content {
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 8px;
  word-break: break-word;
}

.message.user .message-content {
  background: #409eff;
  color: white;
}

.message.assistant .message-content {
  background: #f4f4f5;
  color: #303133;
}

.message-text {
  line-height: 1.6;
}

.message-sources {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid #e4e7ed;
  font-size: 12px;
}

.sources-title {
  color: #909399;
  margin-bottom: 5px;
}

.source-item {
  color: #909399;
  padding: 4px 0;
}

.loading {
  color: #909399;
  font-style: italic;
}

.chat-input {
  padding: 15px 20px;
  border-top: 1px solid #e4e7ed;
  display: flex;
  gap: 10px;
  max-width: 900px;
  margin: 0 auto;
  width: 100%;
}

.chat-input .el-textarea {
  flex: 1;
}

.chat-input .el-button {
  height: fit-content;
}
</style>