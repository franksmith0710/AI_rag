"""
数据库连接模块
负责创建和管理 SQLAlchemy 数据库引擎和会话
只使用 PostgreSQL 生产模式
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from .config import get_settings

settings = get_settings()

# ==================== PostgreSQL 生产模式 ====================
async_engine = create_async_engine(
    settings.postgres_url,
    echo=False,
    pool_pre_ping=True,
    poolclass=NullPool,
    connect_args={
        "timeout": 30,
        "command_timeout": 30,
        "ssl": False
    }
)

async_session_maker = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

sync_engine = create_engine(
    settings.postgres_sync_url,
    echo=False,
    pool_pre_ping=True,
    poolclass=NullPool,
    connect_args={
        "connect_timeout": 30
    }
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
        await conn.run_sync(Base.metadata.create_all)
        from sqlalchemy import text
        await conn.execute(
            text("""INSERT INTO tenants (id, name, created_at, updated_at)
                    VALUES (0, 'Global', NOW(), NOW())
                    ON CONFLICT (id) DO NOTHING""")
        )