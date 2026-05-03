"""
会话管理 API 路由模块
提供会话创建、列表、详情、删除、更新等接口
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from models.schemas import SessionCreate, SessionResponse, SessionListResponse
from services import session_service
from api.auth import get_current_user
from models.schemas import UserResponse
from utils.common import ApiResponse

router = APIRouter(prefix="/sessions", tags=["会话"])


@router.post("", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    创建新会话

    Args:
        session_data: 会话创建数据
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        创建的会话信息
    """
    session = await session_service.create_session(
        db=db,
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        title=session_data.title
    )
    await db.commit()

    return SessionResponse(
        id=session.id,
        tenant_id=session.tenant_id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    skip: int = 0,
    limit: int = 20,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    获取当前用户的所有会话

    Args:
        skip: 跳过条数
        limit: 返回条数
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        会话列表和总数
    """
    return await session_service.get_user_sessions(
        db=db,
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    获取会话详情及消息历史

    Args:
        session_id: 会话 ID
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        会话详情和消息列表
    """
    session = await session_service.get_session_by_id(
        db=db,
        session_id=session_id,
        tenant_id=current_user.tenant_id
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )

    # 获取消息历史
    messages = await session_service.get_session_messages(db, session_id, current_user.tenant_id)

    return SessionResponse(
        id=session.id,
        tenant_id=session.tenant_id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=messages
    )


@router.put("/{session_id}/title")
async def update_session_title(
    session_id: int,
    title: str,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    更新会话标题

    Args:
        session_id: 会话 ID
        title: 新标题
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        更新结果
    """
    session = await session_service.update_session_title(
        db=db,
        session_id=session_id,
        tenant_id=current_user.tenant_id,
        title=title
    )
    await db.commit()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )

    return ApiResponse.success(message="更新成功")


@router.delete("/{session_id}")
async def delete_session(
    session_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    删除会话

    会删除会话及其所有消息

    Args:
        session_id: 会话 ID
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        删除结果
    """
    success = await session_service.delete_session(
        db=db,
        session_id=session_id,
        tenant_id=current_user.tenant_id
    )
    await db.commit()

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )

    return ApiResponse.success(message="删除成功")