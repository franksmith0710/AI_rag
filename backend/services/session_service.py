"""
会话服务模块
负责会话的创建、查询、删除和消息管理
支持会话持久化 (PostgreSQL/SQLite) 和可选的 Redis 缓存
"""
import json
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from models.db_models import Session as SessionModel, Message
from models.schemas import SessionResponse, SessionListResponse, MessageResponse


async def create_session(
    db: AsyncSession,
    tenant_id: int,
    user_id: int,
    title: Optional[str] = None
) -> SessionModel:
    """
    创建新会话

    Args:
        db: 数据库会话
        tenant_id: 租户 ID
        user_id: 用户 ID
        title: 会话标题，默认 "新会话"

    Returns:
        创建的会话对象
    """
    session = SessionModel(
        tenant_id=tenant_id,
        user_id=user_id,
        title=title or "新会话"
    )
    db.add(session)
    await db.flush()
    return session


async def get_session_by_id(
    db: AsyncSession,
    session_id: int,
    tenant_id: int,
    user_id: Optional[int] = None
) -> Optional[SessionModel]:
    """
    根据 ID 获取会话

    Args:
        db: 数据库会话
        session_id: 会话 ID
        tenant_id: 租户 ID (用于权限校验)
        user_id: 用户 ID (可选，用于更严格的权限校验)

    Returns:
        会话对象或 None
    """
    conditions = [
        SessionModel.id == session_id,
        SessionModel.tenant_id == tenant_id
    ]
    if user_id is not None:
        conditions.append(SessionModel.user_id == user_id)

    result = await db.execute(
        select(SessionModel).where(*conditions)
    )
    return result.scalar_one_or_none()


async def get_user_sessions(
    db: AsyncSession,
    tenant_id: int,
    user_id: int,
    skip: int = 0,
    limit: int = 20
) -> SessionListResponse:
    """
    获取用户的所有会话

    Args:
        db: 数据库会话
        tenant_id: 租户 ID
        user_id: 用户 ID
        skip: 跳过条数 (分页)
        limit: 返回条数 (分页)

    Returns:
        会话列表和总数
    """
    result = await db.execute(
        select(SessionModel)
        .where(
            SessionModel.tenant_id == tenant_id,
            SessionModel.user_id == user_id
        )
        .order_by(SessionModel.updated_at.desc())  # 按更新时间降序
        .offset(skip)
        .limit(limit)
    )
    sessions = result.scalars().all()

    # 统计总数
    count_result = await db.execute(
        select(func.count()).select_from(SessionModel).where(
            SessionModel.tenant_id == tenant_id,
            SessionModel.user_id == user_id
        )
    )
    total = count_result.scalar() or 0

    return SessionListResponse(
        total=total,
        items=[SessionResponse(
            id=s.id,
            tenant_id=s.tenant_id,
            user_id=s.user_id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at
        ) for s in sessions]
    )


async def delete_session(
    db: AsyncSession,
    session_id: int,
    tenant_id: int,
    user_id: int
) -> bool:
    """
    删除会话及所有消息

    Args:
        db: 数据库会话
        session_id: 会话 ID
        tenant_id: 租户 ID
        user_id: 用户 ID

    Returns:
        是否删除成功
    """
    session = await get_session_by_id(db, session_id, tenant_id, user_id)
    if not session:
        return False

    # 删除所有消息
    messages_result = await db.execute(
        select(Message).where(Message.session_id == session_id)
    )
    messages = messages_result.scalars().all()
    for msg in messages:
        await db.delete(msg)

    # 删除会话
    await db.delete(session)
    await db.flush()

    # 清除 Redis 缓存 (如果有)
    from core.redis_conn import delete_cached
    cache_key = f"session:{tenant_id}:{session_id}"
    await delete_cached(cache_key)

    return True


async def update_session_title(
    db: AsyncSession,
    session_id: int,
    tenant_id: int,
    user_id: int,
    title: str
) -> Optional[SessionModel]:
    """
    更新会话标题

    Args:
        db: 数据库会话
        session_id: 会话 ID
        tenant_id: 租户 ID
        user_id: 用户 ID
        title: 新标题

    Returns:
        更新后的会话对象
    """
    session = await get_session_by_id(db, session_id, tenant_id, user_id)
    if not session:
        return None

    session.title = title
    await db.flush()
    return session


async def add_message(
    db: AsyncSession,
    session_id: int,
    role: str,
    content: str,
    sources: Optional[list] = None
) -> Message:
    """
    添加消息到会话

    Args:
        db: 数据库会话
        session_id: 会话 ID
        role: 角色 (user/assistant)
        content: 消息内容
        sources: 参考文档来源 (RAG 用)

    Returns:
        创建的消息对象
    """
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources
    )
    db.add(message)

    # 更新会话的更新时间
    result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
    session = result.scalar_one_or_none()
    if session:
        from datetime import datetime
        session.updated_at = datetime.now()

    await db.flush()
    return message


async def get_session_messages(
    db: AsyncSession,
    session_id: int,
    tenant_id: int,
    user_id: int
) -> List[MessageResponse]:
    """
    获取会话的所有消息

    Args:
        db: 数据库会话
        session_id: 会话 ID
        tenant_id: 租户 ID
        user_id: 用户 ID

    Returns:
        消息列表
    """
    session = await get_session_by_id(db, session_id, tenant_id, user_id)
    if not session:
        return []

    messages_result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    )
    messages = messages_result.scalars().all()

    return [MessageResponse(
        id=m.id,
        session_id=m.session_id,
        role=m.role,
        content=m.content,
        sources=m.sources,
        created_at=m.created_at
    ) for m in messages]


async def cache_session_to_redis(session_id: int, tenant_id: int, messages: List[dict]):
    """
    将会话消息缓存到 Redis (可选功能)

    Args:
        session_id: 会话 ID
        tenant_id: 租户 ID
        messages: 消息列表 (dict 格式)
    """
    from core.redis_conn import set_cached
    cache_key = f"session:{tenant_id}:{session_id}"
    cache_value = json.dumps(messages[-10:] if len(messages) > 10 else messages)
    await set_cached(cache_key, cache_value, 3600)


async def get_cached_session_from_redis(session_id: int, tenant_id: int) -> Optional[List[dict]]:
    """
    从 Redis 获取缓存的会话消息

    Args:
        session_id: 会话 ID
        tenant_id: 租户 ID

    Returns:
        缓存的消息列表或 None
    """
    from core.redis_conn import get_cached
    cache_key = f"session:{tenant_id}:{session_id}"
    cached = await get_cached(cache_key)
    if cached:
        return json.loads(cached)
    return None