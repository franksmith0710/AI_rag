# AI RAG 企业知识库系统

> 基于检索增强生成(RAG)技术的智能问答系统，支持文档上传、图片OCR、多租户隔离、GPU加速

## 项目概述

AI RAG 是一个企业级**检索增强生成(RAG)知识库系统**，帮助企业构建私有知识库，实现智能化文档问答。

### 核心功能

| 功能 | 描述 |
|------|------|
| 文档管理 | 支持 PDF/Word/TXT/MD 文档上传、自动分块、向量化存储 |
| 图片OCR | 支持 JPG/PNG/BMP/TIFF 图片文字识别（RapidOCR，本地推理） |
| 智能问答 | 混合检索(向量+BM25) + RRF融合 + 邻居扩展 + 重排序 + DeepSeek LLM 生成 |
| Query 增强 | 一变体查询扩展 + 代词消解（LLM 改写，合并为一次调用），提升检索召回率 |
| 内容去重 | content-hash (MD5) 自动检测重复文档，处理完成后安全清理旧文档 |
| 多租户 | 租户数据完全隔离、支持 RBAC 角色(admin/user/global)、全局文档共享 |
| 会话管理 | 多轮对话、Token 滑动窗口历史管理 + LLM 摘要压缩 + Redis 缓存 |
| 批量操作 | 批量上传 + 并行向量化处理（Semaphore 限制并发防 OOM）|
| GPU 加速 | ONNX Runtime (CUDA) 加速 Embedding 和 Rerank 模型 |
| 流式输出 | SSE 实时流式返回答案（50字符/150ms 攒批推送），前端逐字渲染 |
| 智能标题 | LLM 自动生成会话标题 |

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
│  - SSE 流式解析 (fetch + ReadableStream)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │ /api/*
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI + Uvicorn)                                   │
│  - 端口: 8000                                                  │
│  - JWT 认证                                                    │
│  - 应用启动时预热 GPU 模型 (ONNX)                               │
│  - 线程池 16 workers                                            │
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
                                    │ (ONNX Runtime, CUDA)   │
                                    └─────────────────────────┘
```

## 技术栈

### 后端
- **框架**: FastAPI + Uvicorn
- **数据库**: PostgreSQL + Redis (自动降级内存缓存)
- **ORM**: SQLAlchemy (Async)
- **向量库**: ChromaDB 1.x (PersistentClient, HNSW cosine)
- **全文检索**: BM25 (rank_bm25 + jieba 分词)
- **Embedding**: BGE-M3 (ONNX Runtime, CUDA 加速)
- **Rerank**: BAAI/bge-reranker-v2-m3 (ONNX Runtime, CUDA 加速)
- **LLM**: DeepSeek Chat (langchain_openai.ChatOpenAI / raw openai.AsyncOpenAI)
- **Query 改写**: LLM 驱动 (一次调用完成指代消解 + 变体扩展)
- **OCR**: RapidOCR (ONNX Runtime, 本地推理)
- **其他**: Jieba, tiktoken, Transformers, pypdf, pymupdf

### 前端
- **框架**: Vue 3 (Composition API)
- **UI 组件**: Element Plus
- **状态管理**: Pinia
- **路由**: Vue Router
- **Markdown**: marked

### 部署
- **容器**: Docker (PostgreSQL + Redis + Frontend + Backend)
- **后端**: 原生 uvicorn (dev.ps1 启动)
- **GPU**: NVIDIA GPU (CUDA 12.8, ONNX Runtime)

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| Docker | 20.10+ (仅运行 PostgreSQL + Redis) |
| Python | 3.10+ |
| NVIDIA GPU | 显存 ≥ 4GB |
| CUDA 驱动 | 572.x+ |
| 模型文件 | 存放在 `D:/hf_models` |

### 模型文件准备

确保以下模型已下载到 `D:/hf_models` 目录:

```
D:/hf_models/
├── BAAI/
│   ├── bge-m3/                          # Embedding (PyTorch, tokenizer 需要)
│   ├── bge-m3-onnx/                     # Embedding (ONNX, 主用推理)
│   │   └── bge-m3.onnx
│   ├── bge-reranker-v2-m3/              # Reranker (PyTorch, tokenizer 需要)
│   └── bge-reranker-v2-m3-onnx/         # Reranker (ONNX, 主用推理)
│       └── bge-reranker-v2-m3.onnx
```

GPU 显存限制说明: RTX 3050(4GB) 无法同时加载两个 ONNX 模型到显存。
系统启动时按优先级加载: Reranker 优先 → Embedding 自动降级 CPU。
可在 `dev.ps1` 或 `main.py` 中调整。

### 启动服务

```bash
# 1. 克隆项目后，进入目录
cd AI_rag

# 2. 启动 PostgreSQL + Redis
docker compose up -d postgres redis

# 3. 启动后端 (在单独的终端中)
.\dev.ps1

# 4. 启动前端
cd frontend
npm install
npm run dev
```

### 访问系统

| 服务 | 地址 |
|------|------|
| 前端 Web | http://localhost:3000 (Dev) / http://localhost (Nginx) |
| 后端 API | http://localhost:8000 |
| 健康检查 | http://localhost:8000/health |

### 首次使用

1. 打开前端页面
2. 点击「注册」创建账号 (普通用户会自动创建租户)
3. 登录后，上传文档到知识库
4. 选中文档，点击「处理」，等待向量化完成
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
REDIS_PASSWORD=redis123456

# ==================== DeepSeek LLM ====================
DEEPSEEK_API_KEY=sk-xxx       # 替换为你的 API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# ==================== 模型路径 ====================
EMBEDDING_MODEL_PATH=/models/BAAI/bge-m3
EMBEDDING_ONNX_PATH=/models/BAAI/bge-m3-onnx/bge-m3.onnx
RERANKER_MODEL_PATH=/models/BAAI/bge-reranker-v2-m3
RERANKER_ONNX_PATH=/models/BAAI/bge-reranker-v2-m3-onnx/bge-reranker-v2-m3.onnx
RERANKER_THRESHOLD=0.15

# ==================== LLM 改写配置 ====================
LLM_REWRITE_MODEL=deepseek-chat
QUERY_VARIANT_ENABLED=true
QUERY_VARIANT_COUNT=3

# ==================== JWT ====================
JWT_SECRET_KEY=your-secret-key-change-in-production

# ==================== 可选: LangSmith 监控 ====================
LANGCHAIN_API_KEY=
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=AI_rag
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
            count: 1
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
| POST | /api/documents/upload-batch | 批量上传文档 (≤100MB/文件, ≤200MB/次) |
| POST | /api/documents/process/{id} | 处理文档(分块+向量化)，`?force=true` 强制重处理 |
| POST | /api/documents/process-batch | 批量处理文档 (并行, Semaphore=2 防 OOM) |
| GET | /api/documents | 文档列表 (支持 `?is_global=true/false` 过滤) |
| GET | /api/documents/{id} | 文档详情 |
| DELETE | /api/documents/{id} | 删除文档 (级联删除向量+分块+文件) |
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
| POST | /api/chat | 发送消息 (SSE 流式响应, 自动生成标题) |
| GET | /api/chat/history/{id} | 获取聊天历史 |

## 核心模块说明

### RAG 检索流程

```
用户问题
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. 问候检测                                                  │
│    - 关键词 + 正则匹配短问候语                                │
│    - 命中则直接返回，跳过检索                                 │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Query 改写 + 多变体扩展 (一次 LLM 调用)                   │
│    - 代词消解: "这个政策" → "报销政策的具体内容"                │
│    - 省略补全: 结合历史补全为完整问句                          │
│    - 基于改写结果生成 3 个语义变体，提升召回率                  │
│    - 合并为 rewrite_and_expand()，节省一次 LLM 往返           │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. 混合检索 (多变体并行)                                      │
│    - 向量检索: BGE-M3 (ONNX) + Chroma (余弦相似度)           │
│    - BM25 关键词检索: jieba 分词 + rank_bm25 (带进程缓存)     │
│    - RRF 融合排序 (k=60, 向量权重=0.7 + BM25权重=0.3)        │
│    - 同时检索当前租户 + 全局租户 (tenant_id=0)                │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. 邻居扩展                                                  │
│    - 为每条结果补 ±1 个相邻 chunk (从 PostgreSQL)             │
│    - 保持独立 chunk，不合并                                   │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. 重排序 (BGE Reranker, ONNX CUDA)                         │
│    - Cross-encoder 精细排序，sigmoid 归一化到 [0,1]           │
│    - 过滤低于阈值 (0.15) 的结果                                │
│    - GPU ~0.28s vs CPU ~23s (15 chunks, 80x 加速)            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. LLM 生成答案 (SSE 流式)                                   │
│    - 构建 Prompt (系统提示 + 历史摘要 + 对话历史 + 参考资料)  │
│    - Token 滑动窗口管理历史 (阈值 3000 tokens，从后往前保留)  │
│    - 长对话自动摘要 (LLM 压缩被截断的早期消息, Redis 缓存)     │
│    - 调用 DeepSeek Chat，流式输出                             │
│    - Buffer 策略: 攒够 50 字符或 150ms 后推送                 │
│    - 完成后自动生成会话标题 (LLM)                              │
└──────────────────────────────────────────────────────────────┘
```

### 关键文件

| 文件路径 | 功能描述 |
|----------|----------|
| `backend/main.py` | FastAPI 应用入口，初始化数据库，预热 ONNX 模型，线程池 16 |
| `backend/core/config.py` | 全局配置 (pydantic-settings, lru_cache 单例) |
| `backend/core/chroma_conn.py` | ChromaDB 1.x + BGE-M3 ONNX Embedding (thread-safe) |
| `backend/core/llm_factory.py` | ChatOpenAI 客户端工厂 (集中管理, 替代 5 处实例化) |
| `backend/core/redis_conn.py` | Redis 连接 + 自动降级内存缓存 (1000 条上限) |
| `backend/core/logging_config.py` | 日志配置 |
| `backend/services/rag_service.py` | RAG 核心: 混合检索、邻居扩展、重排序、历史压缩、LLM 流式生成 |
| `backend/services/doc_service.py` | 文档服务: 文本提取、分块、向量化、content-hash 去重 |
| `backend/services/session_service.py` | 会话服务 + Redis 缓存 (消息版本号一致性, 截断 10 条) |
| `backend/services/auth_service.py` | 认证服务 (JWT 签发/验证, 密码哈希) |
| `backend/utils/rerank.py` | BGE Reranker ONNX 重排序 (CUDA 加速, sigmoid) |
| `backend/utils/ocr.py` | RapidOCR 图片文字识别 |
| `backend/utils/splitter.py` | 文档文本分块 (RecursiveCharacterTextSplitter, 中文优化) + jieba BM25 分词 |
| `backend/utils/query_rewrite.py` | Query 改写 + 变体扩展 (合并为一次 LLM 调用) |
| `backend/utils/common.py` | 通用工具 (ApiResponse) |
| `backend/api/auth.py` | 认证 API (注册/登录/me) |
| `backend/api/chat.py` | 问答 API (SSE 流式, 自动标题, Redis 缓存版本校验) |
| `backend/api/document.py` | 文档管理 API (单/批量上传, 单/批量处理, 强制重处理) |
| `backend/api/session.py` | 会话管理 API |
| `backend/init_db.py` | 数据库初始化 (建表 + 默认用户) |
| `backend/scripts/auto_evaluate.py` | 一键自动评估 (生成数据 + 运行评估) |
| `backend/scripts/generate_eval_data.py` (v3) | 评估数据生成 (DeepSeek raw AsyncOpenAI) |
| `backend/scripts/evaluate.py` (v7) | 评估脚本 (余弦相似度 + ROUGE-L, 无 RAGAS 依赖) |
| `frontend/src/views/Chat.vue` | 聊天页面组件 (SSE 解析 + Markdown 渲染) |
| `frontend/src/views/Document.vue` | 文档管理页面 (批量选择/处理) |

## 目录结构

```
AI_rag/
├── backend/                    # 后端服务
│   ├── api/                   # API 路由
│   │   ├── auth.py           # 认证 (注册/登录)
│   │   ├── chat.py           # 问答接口 (SSE)
│   │   ├── document.py       # 文档管理 (单/批量)
│   │   └── session.py        # 会话管理
│   ├── core/                  # 核心模块
│   │   ├── config.py         # 配置管理 (pydantic-settings)
│   │   ├── database.py       # 数据库连接 (asyncpg)
│   │   ├── chroma_conn.py   # ChromaDB 1.x + ONNX Embedding
│   │   ├── llm_factory.py   # ChatOpenAI 客户端工厂
│   │   ├── redis_conn.py    # Redis + 内存缓存降级
│   │   └── logging_config.py # 日志配置
│   ├── models/                # 数据模型
│   │   ├── db_models.py     # SQLAlchemy 模型 (含 content_hash, message_version)
│   │   └── schemas.py       # Pydantic schemas (含批量操作)
│   ├── services/              # 业务逻辑
│   │   ├── rag_service.py   # RAG 核心 (检索 + 重排 + LLM + 历史压缩)
│   │   ├── doc_service.py   # 文档服务 (含 content-hash 去重)
│   │   ├── auth_service.py  # 认证服务
│   │   └── session_service.py # 会话服务 + Redis 缓存
│   ├── utils/                 # 工具函数
│   │   ├── rerank.py        # 重排序 (ONNX CUDA)
│   │   ├── ocr.py           # 图片 OCR (RapidOCR)
│   │   ├── splitter.py      # 文本分块 + jieba BM25 分词
│   │   ├── query_rewrite.py # Query 改写 + 变体扩展 (LLM)
│   │   └── common.py        # 通用工具
│   ├── scripts/               # 脚本工具
│   │   ├── evaluate.py      # 评估脚本 (v7, 手动指标)
│   │   ├── generate_eval_data.py # 评估数据生成 (v3)
│   │   ├── auto_evaluate.py # 一键自动评估
│   │   ├── reindex_prep.py  # 索引重建准备
│   │   └── test_data.json   # 评估测试数据
│   ├── main.py               # 应用入口 (lifespan, GPU 预热)
│   ├── init_db.py            # 数据库初始化
│   ├── Dockerfile            # 后端镜像
│   ├── uploads/              # 上传文件目录
│   └── vector_store/         # Chroma 数据目录
│
├── frontend/                  # 前端服务
│   ├── src/
│   │   ├── api/              # API 调用
│   │   ├── components/       # 通用组件
│   │   ├── views/            # 页面组件
│   │   │   ├── Login.vue     # 登录页
│   │   │   ├── Register.vue  # 注册页
│   │   │   ├── Chat.vue      # 聊天页 (SSE + 欢迎屏)
│   │   │   └── Document.vue  # 文档管理页 (批量勾选)
│   │   ├── stores/           # Pinia 状态
│   │   ├── router.js         # 路由配置
│   │   ├── App.vue           # 根组件
│   │   └── main.js           # 入口
│   ├── nginx.conf            # Nginx 配置
│   ├── Dockerfile            # 前端镜像
│   └── package.json          # Node 依赖
│
├── docker-compose.yml         # Docker 编排 (PostgreSQL + Redis + Backend + Frontend)
├── dev.ps1                    # 开发启动脚本 (ONNX GPU 路径)
├── .env                       # 环境变量 (敏感信息)
├── .env.example               # 环境变量模板
└── README.md                  # 项目文档
```

## 常见问题

### Q: 模型文件路径如何修改?

修改 `.env` 中的模型路径:
```env
EMBEDDING_MODEL_PATH=/models/BAAI/bge-m3
EMBEDDING_ONNX_PATH=/models/BAAI/bge-m3-onnx/bge-m3.onnx
RERANKER_MODEL_PATH=/models/BAAI/bge-reranker-v2-m3
RERANKER_ONNX_PATH=/models/BAAI/bge-reranker-v2-m3-onnx/bge-reranker-v2-m3.onnx
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

3. 启动日志中会显示:
```
Reranker ONNX 模型加载完成, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
```

注意: RTX 3050(4GB VRAM) 无法同时加载 Embedding 和 Reranker 两个模型到显卡。
系统按优先级启动: Reranker 优先使用 GPU，Embedding 自动降级到 CPU。
如需交换，修改 `main.py` 中预热顺序。

### Q: 上传文档后需要做什么?

1. 点击文档的「处理」按钮（支持批量勾选后点击「批量处理」）
2. 系统会自动:
   - 提取文本内容（PDF 含 OCR 兜底，图片用 RapidOCR）
   - 计算 content-hash，检测并标记重复文档
   - 分块（默认 650 字符/块，重叠 100 字符）
   - 向量化存储到 Chroma (ONNX 推理)
3. 处理完成后即可开始问答

### Q: 重复文档如何处理?

系统有双重去重机制:

1. **同名覆盖**: 同租户下上传相同文件名的文档，直接覆盖（保留旧文件路径记录）
2. **内容哈希去重**: 处理文档时计算 `md5(text_content.strip())`，检测到相同内容的旧文档后，先处理新文档，成功后再安全清理旧文档的向量、分块和文件

### Q: 如何实现多租户隔离?

- 每个租户拥有独立的 Chroma collection (`tenant_{id}_documents`)
- 文档查询时同时检索「当前租户」和「全局租户(tenant_id=0)」
- admin 可上传全局共享文档，普通用户只能访问自己租户
- ChromaDB 1.x 通过 AdminClient 自动创建 `default_tenant` 和 `default_database`

### Q: 如何运行评估?

```bash
cd backend

# 一键自动评估（生成测试数据 + 运行评估）
python scripts/auto_evaluate.py

# 或分步执行：
# 1. 生成测试数据（从数据库 chunks 自动提取）
python scripts/generate_eval_data.py

# 2. 运行评估
python scripts/evaluate.py
```

评估指标 (纯本地，不依赖 RAGAS):
- **检索相关性**: 问题与检索到的 chunk 的余弦相似度 (BGE-M3 embedding cosine)，取 max/avg
- **QA 语义相似度**: 问题与 ground_truth 的语义相似度 (BGE-M3 embedding cosine)
- **答案接地性**: ground_truth 与检索结果的最长公共子序列 F1 (ROUGE-L)

关键注意事项:
- 评估脚本必须在 `get_settings()` 之前设置 `os.environ`，否则 `.env` 路径错误
- 使用 raw `openai.AsyncOpenAI` 而非 LangChain，避免 LangSmith 干扰
- 模型名使用 `deepseek-chat`（非 `deepseek-v4-flash`，需特定权限）
- 结果保存在 `backend/scripts/eval_report.json`

## 开发指南

### 本地开发

```bash
# 1. 启动 PostgreSQL + Redis
docker compose up -d postgres redis

# 2. 启动后端 (自动设置本地路径和 CUDA 环境)
.\dev.ps1

# 3. 启动前端 (另一个终端)
cd frontend
npm install
npm run dev
```

`dev.ps1` 自动覆写 Docker 环境变量为本地路径:
- PostgreSQL → `localhost:5433`
- ONNX 模型 → `D:\hf_models\BAAI\...`
- Chroma 持久化 → `D:\School\AI_rag\backend\vector_store\chroma`
- CUDA → `D:\miniconda3\envs\py312\Lib\site-packages\torch\lib`

### 查看日志

```bash
# 后端日志
Get-Content backend\logs\*.log -Tail 50

# Docker 日志
docker compose logs -f postgres redis
```

## License

MIT License
