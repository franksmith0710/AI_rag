"""
配置管理模块
负责加载和管理所有配置项，支持从 .env 文件读取环境变量
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    全局配置类
    使用 pydantic-settings 自动从环境变量加载配置
    """

    # ==================== LLM 配置 ====================
    deepseek_api_key: str = ""  # 从 .env 读取
    deepseek_base_url: str = ""  # 从 .env 读取
    deepseek_model: str = ""  # 从 .env 读取

    # ==================== 数据库配置 ====================

    # PostgreSQL 配置 (生产环境使用)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "rag_db"

    # ==================== Redis 配置 ====================
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # ==================== 向量库配置 ====================
    vector_store: str = "chroma"  # 向量库模式: chroma(开发) / milvus(生产)
    chroma_persist_dir: str = "./vector_store/chroma"  # Chroma 数据持久化目录

    # Milvus 配置 (生产环境使用)
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # ==================== 文件存储配置 ====================
    storage_mode: str = "local"  # 文件存储模式: local(开发) / minio(生产)

    # MinIO 配置 (生产环境使用)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-documents"

    

    # ==================== JWT 配置 ====================
    jwt_secret_key: str = "your-secret-key-change-in-production"  # JWT 密钥
    jwt_algorithm: str = "HS256"  # JWT 算法
    jwt_expire_minutes: int = 1440  # Token 过期时间(分钟)

    # ==================== 文件上传配置 ====================
    upload_dir: str = "./uploads"  # 上传文件存储目录

    # ==================== LangSmith 配置 ====================
    langchain_api_key: str = ""  # 从 .env 读取（LANGCHAIN_API_KEY）
    langchain_endpoint: str = ""  # 从 .env 读取（LANGCHAIN_ENDPOINT）
    langchain_project: str = "AI_rag"  # 从 .env 读取（LANGCHAIN_PROJECT）
    langchain_tracing_v2: bool = True  # 从 .env 读取（LANGCHAIN_TRACING_V2）

    # ==================== RAG 配置 ====================
    reranker_model_path: str = "/models/BAAI/bge-reranker-v2-m3"
    reranker_onnx_path: str = "/models/BAAI/bge-reranker-v2-m3-onnx/bge-reranker-v2-m3.onnx"
    embedding_model_path: str = "/models/BAAI/bge-m3"
    embedding_onnx_path: str = "/models/BAAI/bge-m3-onnx/bge-m3.onnx"
    reranker_threshold: float = 0.1
    llm_rewrite_model: str = "deepseek-chat"
    query_variant_enabled: bool = True
    query_variant_count: int = 3
    redis_enabled: bool = True

    @property
    def postgres_url(self) -> str:
        """返回 PostgreSQL 异步连接 URL (用于 asyncpg)"""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_sync_url(self) -> str:
        """返回 PostgreSQL 同步连接 URL (用于 SQLAlchemy)"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}?sslmode=disable"

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    使用 lru_cache 缓存配置对象，整个应用共享同一个配置实例
    """
    return Settings()