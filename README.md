# AI RAG 企业知识库系统

> 基于检索增强生成(RAG)技术的智能问答系统，支持文档上传、多租户隔离、GPU 加速

## 项目概述

AI RAG 是一个企业级**检索增强生成(RAG)知识库系统**，帮助企业构建私有知识库，实现智能化文档问答。

### 核心功能

| 功能 | 描述 |
|------|------|
| 📄 文档管理 | 支持 PDF/Word/TXT/MD 文档上传、自动分块、向量化存储 |
| 💬 智能问答 | 混合检索(向量+BM25) + 重排序 + DeepSeek LLM 生成答案 |
| 👥 多租户 | 租户数据完全隔离、支持 RBAC 角色(admin/user) |
| 💾 会话管理 | 多轮对话、历史记录持久化 |
| 🚀 GPU 加速 | 支持 NVIDIA GPU 加速 Embedding 和 Rerank 模型 |

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户请求                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (Vue3 + Element Plus + Nginx)                        │
│  - 端口: 80 (HTTP) / 3000 (Dev)                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │ /api/*
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI + Uvicorn)                                   │
│  - 端口: 8000                                                  │
│  - JWT 认证                                                    │
│  - 应用启动时预热 GPU 模型                                     │
└────────┬────────────────────┬───────────────────┬──────────────┘
         │                    │                   │
         ▼                    ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  PostgreSQL     │  │  Redis          │  │  Chroma DB     │
│  (主数据库)     │  │  (缓存)          │  │  (向量存储)     │
│  端口: 5433     │  │  端口: 6379      │  │                │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                                                  │
                                                  ▼
                                   ┌─────────────────────────┐
                                   │   AI Models (GPU)       │
                                   │ - BGE-M3 (Embedding)   │
                                   │ - bge-reranker-v2-m3   │
                                   │ - DeepSeek LLM         │
                                   └─────────────────────────┘
```

## 技术栈

### 后端
- **框架**: FastAPI + Uvicorn
- **数据库**: PostgreSQL + Redis
- **ORM**: SQLAlchemy (Async)
- **RAG**: LangChain + ChromaDB
- **Embedding**: BGE-M3 (本地模型)
- **Rerank**: BAAI/bge-reranker-v2-m3
- **LLM**: DeepSeek Chat (deepseek-chat)
- **其他**: Jieba (中文分词), BM25, Transformers, PyTorch

### 前端
- **框架**: Vue 3 (Composition API)
- **UI 组件**: Element Plus
- **状态管理**: Pinia
- **路由**: Vue Router
- **Markdown**: marked

### 部署
- **容器**: Docker + Docker Compose
- **Web 服务器**: Nginx (Frontend 反向代理)
- **GPU**: NVIDIA GPU (CUDA)

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| Docker | 20.10+ |
| Docker Compose | 2.0+ |
| NVIDIA GPU | 显存 ≥ 4GB |
| CUDA 驱动 | 572.x+ |
| 模型文件 | 存放在 `D:/hf_models` |

### 模型文件准备

确保以下模型已下载到 `D:/hf_models` 目录:

```
D:/hf_models/
├── BAAI/
│   ├── bge-m3/                  # Embedding 模型
│   └── bge-reranker-v2-m3/     # Reranker 模型
```

### 启动服务

```bash
# 1. 克隆项目后，进入目录
cd AI_rag

# 2. 启动所有服务
docker-compose up -d

# 3. 查看服务状态
docker-compose ps

# 4. 查看后端日志 (确认 GPU 加载成功)
docker-compose logs -f backend
```

启动成功后，日志应显示:

```
设备: cuda
Embedding 模型加载完成
Reranker 模型加载完成
```

### 访问系统

| 服务 | 地址 |
|------|------|
| 前端 Web | http://localhost |
| 后端 API | http://localhost:8000 |
| 健康检查 | http://localhost:8000/health |

### 首次使用

1. 打开 http://localhost
2. 点击「注册」创建账号 (普通用户会自动创建租户)
3. 登录后，上传文档到知识库
4. 点击「处理」文档，等待向量化完成
5. 返回聊天页面，创建会话，开始问答

## 配置说明

### 环境变量 (.env)

```env
# ==================== 数据库 ====================
POSTGRES_HOST=postgres        # Docker 服务名
POSTGRES_PORT=5432            # 容器内端口
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=rag_db

REDIS_HOST=redis              # Docker 服务名
REDIS_PORT=6379

# ==================== DeepSeek LLM ====================
DEEPSEEK_API_KEY=sk-xxx       # 替换为你的 API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# ==================== 模型路径 (Docker 内) ====================
EMBEDDING_MODEL_PATH=/models/BAAI/bge-m3
RERANKER_MODEL_PATH=/models/BAAI/bge-reranker-v2-m3
RERANKER_THRESHOLD=0.1

# ==================== JWT ====================
JWT_SECRET_KEY=your-secret-key-change-in-production

# ==================== LangSmith (可选) ====================
LANGCHAIN_API_KEY=lsv2_xxx
LANGCHAIN_TRACING_V2=true
```

### Docker GPU 配置

在 `docker-compose.yml` 中已配置 GPU 访问:

```yaml
backend:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
```

## API 接口文档

### 认证模块

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /api/auth/register | 用户注册 |
| POST | /api/auth/login | 用户登录 |
| GET | /api/auth/me | 获取当前用户信息 |

### 文档模块

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /api/documents/upload | 上传文档 |
| POST | /api/documents/process/{id} | 处理文档(分块+向量化) |
| GET | /api/documents | 文档列表 |
| GET | /api/documents/{id} | 文档详情 |
| DELETE | /api/documents/{id} | 删除文档 |
| GET | /api/documents/{id}/chunks | 查看文档分块 |

### 会话模块

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /api/sessions | 创建新会话 |
| GET | /api/sessions | 会话列表 |
| DELETE | /api/sessions/{id} | 删除会话 |

### 问答模块

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /api/chat | 发送消息 (流式响应) |
| GET | /api/chat/history/{id} | 获取聊天历史 |

## 核心模块说明

### RAG 检索流程

```
用户问题
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Query 改写 (基于历史对话)                                  │
│    - 使用 DeepSeek 理解上下文                                │
│    - 优化搜索关键词                                          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. 混合检索                                                  │
│    - 向量检索 (BGE-M3 + Chroma)                              │
│    - BM25 关键词检索                                         │
│    - RRF 融合排序                                            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. 重排序 (BGE Reranker)                                    │
│    - 使用 cross-encoder 精细排序                             │
│    - 过滤低于阈值的结果 (默认 0.1)                            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. LLM 生成答案                                              │
│    - 构建 Prompt (系统提示 + 上下文 + 参考资料)              │
│    - 调用 DeepSeek Chat                                      │
│    - 流式输出答案                                            │
└──────────────────────────────────────────────────────────────┘
```

### 关键文件

| 文件路径 | 功能描述 |
|----------|----------|
| `backend/main.py` | FastAPI 应用入口，初始化数据库，预热模型 |
| `backend/services/rag_service.py` | RAG 核心: 检索、重排序、LLM 生成 |
| `backend/core/chroma_conn.py` | Chroma 向量库 + BGE-M3 Embedding |
| `backend/utils/rerank.py` | BGE Reranker 重排序 (GPU 加速) |
| `backend/utils/splitter.py` | 文档文本分块 (RecursiveCharacterTextSplitter) |
| `backend/api/chat.py` | 问答 API (流式响应 SSE) |
| `backend/api/document.py` | 文档管理 API |
| `backend/models/db_models.py` | SQLAlchemy 数据模型 |
| `frontend/src/views/Chat.vue` | 聊天页面组件 |
| `frontend/nginx.conf` | Nginx 配置 (/api 代理到后端) |

## 目录结构

```
AI_rag/
├── backend/                    # 后端服务
│   ├── api/                   # API 路由
│   │   ├── auth.py           # 认证 (注册/登录)
│   │   ├── chat.py           # 问答接口
│   │   ├── document.py       # 文档管理
│   │   └── session.py        # 会话管理
│   ├── core/                  # 核心模块
│   │   ├── config.py         # 配置管理
│   │   ├── database.py       # 数据库连接
│   │   ├── chroma_conn.py   # Chroma 向量库
│   │   └── redis_conn.py    # Redis 连接
│   ├── models/                # 数据模型
│   │   ├── db_models.py     # SQLAlchemy 模型
│   │   └── schemas.py       # Pydantic schemas
│   ├── services/              # 业务逻辑
│   │   ├── rag_service.py   # RAG 核心
│   │   ├── doc_service.py   # 文档服务
│   │   ├── auth_service.py  # 认证服务
│   │   └── session_service.py # 会话服务
│   ├── utils/                 # 工具函数
│   │   ├── rerank.py        # 重排序
│   │   ├── splitter.py      # 文本分块
│   │   ├── query_rewrite.py # Query 改写
│   │   └── common.py        # 通用工具
│   ├── main.py               # 应用入口
│   ├── requirements.txt      # Python 依赖
│   ├── Dockerfile           # 后端镜像
│   ├── uploads/              # 上传文件目录
│   └── vector_store/         # Chroma 数据目录
│
├── frontend/                  # 前端服务
│   ├── src/
│   │   ├── api/              # API 调用
│   │   ├── views/            # 页面组件
│   │   │   ├── Login.vue     # 登录页
│   │   │   ├── Register.vue  # 注册页
│   │   │   ├── Chat.vue      # 聊天页
│   │   │   └── Document.vue  # 文档管理页
│   │   ├── stores/           # Pinia 状态
│   │   └── router.js         # 路由配置
│   ├── nginx.conf            # Nginx 配置
│   ├── Dockerfile            # 前端镜像
│   └── package.json          # Node 依赖
│
├── docker-compose.yml         # Docker 编排
├── .env                       # 环境变量
└── README.md                  # 项目文档
```

## 常见问题

### Q: 模型文件路径如何修改?

修改 `.env` 中的模型路径:
```env
EMBEDDING_MODEL_PATH=/models/BAAI/bge-m3
RERANKER_MODEL_PATH=/models/BAAI/bge-reranker-v2-m3
```

同时修改 `docker-compose.yml` 中的挂载路径:
```yaml
volumes:
  - D:/你的模型路径:/models
```

### Q: 如何确认 GPU 正在被使用?

1. 查看后端日志:
```bash
docker-compose logs backend | grep "设备"
```
应显示 `设备: cuda`

2. 查看 GPU 使用情况:
```bash
nvidia-smi
```

### Q: 上传文档后需要做什么?

1. 点击文档的「处理」按钮
2. 系统会自动:
   - 提取文本内容
   - 分块 (默认 650 字符/块)
   - 向量化存储到 Chroma
3. 处理完成后即可开始问答

### Q: 如何实现多租户隔离?

- 每个租户拥有独立的 Chroma collection (`tenant_{id}_documents`)
- 文档查询时同时检索「当前租户」和「全局租户(tenant_id=0)」
- admin 可上传全局共享文档，普通用户只能访问自己租户

## 开发指南

### 本地开发 (非 Docker)

```bash
# 后端
cd backend
pip install -r requirements.txt
python main.py

# 前端
cd frontend
npm install
npm run dev
```

### 重新构建镜像

```bash
docker-compose build --no-cache
```

### 查看日志

```bash
# 所有服务
docker-compose logs -f

# 指定服务
docker-compose logs -f backend
docker-compose logs -f frontend
```

## License

MIT License