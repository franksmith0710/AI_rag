"""
FastAPI 应用入口
负责应用初始化、中间件配置、路由注册
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
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

    关闭时:
    - 关闭 Redis 连接
    """
    await init_db()
    logger.info("Database initialized")
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,  # 允许携带凭证
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
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
        host="0.0.0.0",  # 监听所有网络接口
        port=8000,      # 端口 8000
        reload=True     # 开发模式热重载
    )