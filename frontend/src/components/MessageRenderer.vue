<template>
  <div class="message-renderer">
    <!-- 纯文本显示（无底部分割线） -->
    <div v-if="data.rawText" class="raw-text">
      {{ data.rawText }}
    </div>
    
    <!-- JSON 结构化回答 -->
    <div v-else-if="data.answer" class="answer-summary">
      {{ data.answer }}
    </div>
    
    <div v-for="(section, index) in data.sections" :key="index" class="section">
      <div class="section-title">{{ section.title }}</div>
      
      <!-- text 类型 -->
      <div v-if="section.type === 'text'" class="section-content">
        {{ section.content }}
      </div>
      
      <!-- list 类型 -->
      <ul v-else-if="section.type === 'list'" class="section-list">
        <li v-for="(item, idx) in section.items" :key="idx">{{ item }}</li>
      </ul>
      
      <!-- table 类型 -->
      <div v-else-if="section.type === 'table'" class="section-table">
        <table>
          <thead>
            <tr>
              <th v-for="(header, idx) in section.headers" :key="idx">{{ header }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, idx) in section.rows" :key="idx">
              <td v-for="(cell, cidx) in row" :key="cidx">{{ cell }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  data: {
    type: Object,
    required: true,
    default: () => ({})
  }
})
</script>

<style scoped>
.message-renderer {
  line-height: 1.6;
}

.raw-text {
  white-space: pre-wrap;
  word-break: break-word;
}

.answer-summary {
  font-weight: bold;
  font-size: 16px;
  color: #303133;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #ebeef5;
}

.section {
  margin-bottom: 16px;
}

.section-title {
  font-weight: bold;
  font-size: 15px;
  color: #409eff;
  margin-bottom: 8px;
}

.section-content {
  color: #606266;
  line-height: 1.8;
  white-space: pre-wrap;
}

.section-list {
  margin: 0;
  padding-left: 20px;
  color: #606266;
}

.section-list li {
  margin-bottom: 4px;
  line-height: 1.6;
}

.section-table {
  overflow-x: auto;
}

.section-table table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.section-table th,
.section-table td {
  border: 1px solid #dcdfe6;
  padding: 8px 12px;
  text-align: left;
}

.section-table th {
  background: #f5f7fa;
  font-weight: bold;
  color: #303133;
}

.section-table td {
  color: #606266;
}
</style>