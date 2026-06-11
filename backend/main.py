"""
FastAPI 应用入口
负责应用初始化、中间件配置、路由注册
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import uvicorn

from core.config import get_settings
from core.database import init_db
from core.redis_conn import close_redis
from core.logging_config import setup_logging
from api import auth, document, chat, session

settings = get_settings()
logger = setup_logging("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时:
    - 初始化数据库表结构
    - 预热 GPU 模型

    关闭时:
    - 关闭 Redis 连接
    """
    await init_db()
    logger.info("Database initialized")

    # JWT 密钥安全检查
    if settings.jwt_secret_key == "your-secret-key-change-in-production":
        logger.error("❌  JWT 密钥使用默认值，请先修改 JWT_SECRET_KEY")
        sys.exit(1)

    # 增大线程池，提升并行处理能力
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=16))
    logger.info("线程池已扩大到 16 workers")

    logger.info("预热 Jieba 分词器...")
    import jieba
    jieba.initialize()
    logger.info("Jieba 预热完成")

    yield

    await close_redis()
    logger.info("Redis closed")


# 创建 FastAPI 应用
app = FastAPI(
    title="企业级RAG知识库系统",
    description="支持文档上传、检索、多轮对话的企业知识库系统",
    version="1.0.0",
    lifespan=lifespan
)

# ==================== 中间件 ====================

# CORS 中间件：允许跨域访问
cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 路由注册 ====================

app.include_router(auth.router, prefix="/api")
app.include_router(document.router, prefix="/api")
app.include_router(session.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

# ==================== 根路由 ====================

@app.get("/")
async def root():
    """根路径，返回应用状态"""
    return {"message": "RAG知识库系统API", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


# ==================== 启动配置 ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )