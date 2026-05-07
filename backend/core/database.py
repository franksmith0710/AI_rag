"""
数据库连接模块
负责创建和管理 SQLAlchemy 数据库引擎和会话
支持 SQLite(开发) 和 PostgreSQL(生产) 两种模式
"""
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from .config import get_settings

settings = get_settings()

# 根据配置选择数据库模式
if settings.db_mode == "sqlite":
    # ==================== SQLite 开发模式 ====================
    # 使用 aiosqlite 异步驱动，无需额外安装数据库服务

    # 确保数据库目录存在
    db_path = settings.sqlite_db_path
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 创建异步引擎
    async_engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}?timezone=Asia/Shanghai",
        echo=False,
        pool_pre_ping=True
    )

    # 创建异步会话工厂
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False  # 提交后不自动过期对象
    )

    # 创建同步引擎
    sync_engine = create_engine(
        f"sqlite:///{db_path}?timezone=Asia/Shanghai",
        echo=False,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False}
    )
else:
    # ==================== PostgreSQL 生产模式 ====================
    # 使用 asyncpg 异步驱动

    async_engine = create_async_engine(
        settings.postgres_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,  # 连接池大小
        max_overflow=20  # 最大溢出连接数
    )

    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    sync_engine = create_engine(
        settings.postgres_sync_url,
        echo=False,
        pool_pre_ping=True
    )

# 创建 ORM 基类，所有模型类继承此基类
Base = declarative_base()


async def get_db() -> AsyncSession:
    """
    数据库会话依赖注入函数
    用于 FastAPI 的 Depends(get_db)，自动管理会话生命周期

    使用示例:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_maker() as session:
        try:
            yield session  # 返回会话供路由使用
            await session.commit()  # 成功后自动提交
        except Exception:
            await session.rollback()  # 失败时回滚
            raise
        finally:
            await session.close()  # 关闭会话


async def init_db():
    """
    初始化数据库表
    在应用启动时调用，创建所有表结构
    """
    async with async_engine.begin() as conn:
        # run_sync 用于在异步上下文中运行同步操作
        await conn.run_sync(Base.metadata.create_all)