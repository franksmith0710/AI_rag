"""
认证服务模块
负责用户注册、登录、Token 生成和验证
使用 JWT 进行无状态认证，bcrypt 进行密码哈希
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.db_models import User, Tenant
from models.schemas import UserCreate
from core.config import get_settings

settings = get_settings()

# 密码哈希上下文，使用 bcrypt 算法
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码是否正确

    Args:
        plain_password: 明文密码
        hashed_password: 数据库存储的哈希密码

    Returns:
        密码是否匹配
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    对密码进行哈希

    Args:
        password: 明文密码

    Returns:
        哈希后的密码字符串
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建 JWT Access Token

    Args:
        data: 要编码的数据 (如 {"sub": user_id, "tenant_id": tenant_id})
        expires_delta: 过期时间增量，默认使用配置中的时间

    Returns:
        JWT Token 字符串
    """
    to_encode = data.copy()
    # 设置过期时间
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.jwt_expire_minutes))
    to_encode.update({"exp": expire})

    # 使用 HS256 算法签名
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Optional[dict]:
    """
    解码并验证 JWT Token

    Args:
        token: JWT Token 字符串

    Returns:
        解码后的 payload，或 Token 无效时返回 None
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError:
        return None


async def create_tenant(db: AsyncSession, name: str) -> Tenant:
    """
    创建新租户

    Args:
        db: 数据库会话
        name: 租户名称

    Returns:
        创建的租户对象
    """
    tenant = Tenant(name=name)
    db.add(tenant)
    await db.flush()  # 获取生成的 ID
    return tenant


async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    """
    创建新用户

    Args:
        db: 数据库会话
        user_data: 用户创建数据

    Returns:
        创建的用户对象
    """
    password_hash = get_password_hash(user_data.password)
    user = User(
        tenant_id=user_data.tenant_id,
        username=user_data.username,
        password_hash=password_hash,
        role=user_data.role
    )
    db.add(user)
    await db.flush()
    return user


async def authenticate_user(db: AsyncSession, username: str, password: str) -> Optional[User]:
    """
    用户认证 (登录)

    Args:
        db: 数据库会话
        username: 用户名
        password: 明文密码

    Returns:
        认证成功的用户对象，或认证失败返回 None
    """
    # 查找用户
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    # 验证密码
    if user and verify_password(password, user.password_hash):
        return user
    return None


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """
    根据 ID 获取用户

    Args:
        db: 数据库会话
        user_id: 用户 ID

    Returns:
        用户对象，或不存在返回 None
    """
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """
    根据用户名获取用户

    Args:
        db: 数据库会话
        username: 用户名

    Returns:
        用户对象，或不存在返回 None
    """
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()