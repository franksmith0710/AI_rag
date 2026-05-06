"""
配置管理模块
负责加载和管理所有配置项，支持从 .env 文件读取环境变量
"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    全局配置类
    使用 pydantic-settings 自动从环境变量加载配置
    """

    # ==================== LLM 配置 ====================
    deepseek_api_key: str = ""  # DeepSeek API 密钥
    deepseek_base_url: str = "https://api.deepseek.com"  # DeepSeek API 地址
    deepseek_model: str = "deepseek-chat"  # 使用的模型名称

    # ==================== 数据库配置 ====================
    db_mode: str = "sqlite"  # 数据库模式: sqlite(开发) / postgresql(生产)
    sqlite_db_path: str = "./data/rag.db"  # SQLite 数据库文件路径

    # PostgreSQL 配置 (生产环境使用)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "rag_db"

    # ==================== Redis 配置 ====================
    redis_host: str = "localhost"
    redis_port: int = 6379

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

    # ==================== 模型配置 ====================
    bge_model_path: str = "./models/bge-m3"  # BGE 向量模型路径
    rerank_model_path: str = "./models/bge-reranker-base"  # BGE 重排序模型路径

    # ==================== Ollama 配置 ====================
    ollama_host: str = ""  # Ollama 服务地址，优先使用环境变量，为空时使用默认值
    ollama_embed_model: str = "bge-m3"  # Ollama Embedding 模型名称

    # ==================== JWT 配置 ====================
    jwt_secret_key: str = "your-secret-key-change-in-production"  # JWT 密钥
    jwt_algorithm: str = "HS256"  # JWT 算法
    jwt_expire_minutes: int = 1440  # Token 过期时间(分钟)

    # ==================== 文件上传配置 ====================
    upload_dir: str = "./uploads"  # 上传文件存储目录

    @property
    def sqlite_url(self) -> str:
        """返回 SQLite 数据库连接 URL"""
        return f"sqlite:///{self.sqlite_db_path}"

    @property
    def postgres_url(self) -> str:
        """返回 PostgreSQL 异步连接 URL (用于 aiosqlite/asyncpg)"""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_sync_url(self) -> str:
        """返回 PostgreSQL 同步连接 URL (用于 SQLAlchemy)"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    class Config:
        env_file = "../.env"  # 从项目根目录的 .env 文件加载环境变量
        extra = "ignore"  # 忽略额外字段


# 使用 lru_cache 缓存配置实例，避免重复读取
@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    使用 lru_cache 缓存配置对象，整个应用共享同一个配置实例
    """
    return Settings()