# 企业级RAG知识库系统

基于 LangChain 实现的企业级私有知识库 RAG 系统，支持文档上传、检索、多轮对话。

## 功能特性

- **多租户支持**：支持多企业/部门独立管理
- **RBAC权限**：管理员/普通用户角色控制
- **文档管理**：支持 PDF/Word/TXT 文档上传、解析、向量化
- **混合检索**：向量检索 + BM25 多路召回 + Rerank 重排序
- **多轮对话**：会话持久化、上下文记忆、指代消解
- **答案溯源**：生成答案时附带参考文档来源
- **Docker部署**：一键启动所有服务

## 技术架构

| 组件 | 技术选型 |
|------|----------|
| 前端 | Vue3 + Element Plus |
| 后端 | FastAPI + LangChain |
| LLM | DeepSeek API |
| Embedding | BGE-m3 (本地) |
| Rerank | BGE-reranker-base (本地) |
| 向量库 | Milvus |
| 业务DB | PostgreSQL |
| 缓存 | Redis |
| 对象存储 | MinIO |

## 快速开始

### 1. 启动基础设施

```bash
docker-compose up -d
```

等待所有服务启动完成（约1-2分钟）。

### 2. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 3. 初始化数据库

```bash
cd backend
python init_db.py
```

### 4. 下载模型

确保模型文件已下载到指定目录：
- BGE-m3: `./models/bge-m3`
- BGE-reranker-base: `./models/bge-reranker-base`

如无模型，可通过 HuggingFace 自动下载（首次运行时会自动下载）。

### 5. 启动后端服务

```bash
cd backend
python main.py
```

服务将在 http://localhost:8000 启动。

### 6. 安装并启动前端

```bash
cd frontend
npm install
npm run dev
```

前端将在 http://localhost:3000 启动。

### 7. 登录系统

默认管理员账号：
- 用户名：`admin`
- 密码：`admin123`

## API 文档

服务启动后访问：
- FastAPI docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 项目结构

```
AI_rag/
├── backend/
│   ├── api/              # API路由
│   ├── services/         # 业务逻辑
│   ├── models/          # 数据模型
│   ├── utils/           # 工具函数
│   ├── core/            # 核心配置
│   ├── main.py          # 入口文件
│   └── requirements.txt # 依赖
├── frontend/
│   ├── src/
│   │   ├── views/       # 页面组件
│   │   ├── api/         # 接口封装
│   │   └── stores/      # 状态管理
│   └── package.json
├── docker-compose.yml    # 基础设施
├── .env                 # 环境配置
└── README.md
```

## 使用流程

1. **上传文档**：进入知识库管理页面上传 PDF/Word/TXT 文档
2. **处理文档**：点击"处理"按钮进行分块、向量化
3. **创建会话**：在对话页面点击"新建会话"
4. **提问**：开始问答，系统会检索相关文档生成答案

## 配置说明

修改 `.env` 文件调整配置：

- `DEEPSEEK_API_KEY`: DeepSeek API 密钥
- `POSTGRES_*`: PostgreSQL 连接配置
- `REDIS_*`: Redis 连接配置
- `MILVUS_*`: Milvus 连接配置
- `BGE_MODEL_PATH`: BGE 模型路径
- `RERANK_MODEL_PATH`: Rerank 模型路径

## 注意事项

1. 首次运行会自动下载 BGE 模型（约 1-2GB），请确保网络畅通
2. 确保 Docker 服务正常运行
3. 如遇端口冲突，请修改 docker-compose.yml 中的端口映射