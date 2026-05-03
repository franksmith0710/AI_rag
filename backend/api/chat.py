"""
问答 API 路由模块
提供聊天接口和历史消息查询
使用 RAG 技术进行问答
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from models.schemas import ChatRequest, ChatResponse
from services import session_service, rag_service
from api.auth import get_current_user
from models.schemas import UserResponse
from utils.common import ApiResponse

router = APIRouter(prefix="/chat", tags=["问答"])


@router.post("", response_model=ChatResponse)
async def chat(
    chat_data: ChatRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    发送消息进行问答

    流程:
    1. 验证会话存在且属于当前用户
    2. 保存用户消息
    3. 执行 RAG 检索和生成
    4. 保存 AI 回复

    Args:
        chat_data: 聊天请求 (session_id, message)
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        AI 回复和参考文档来源
    """
    # 验证会话
    session = await session_service.get_session_by_id(
        db=db,
        session_id=chat_data.session_id,
        tenant_id=current_user.tenant_id
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )

    # 保存用户消息
    user_message = await session_service.add_message(
        db=db,
        session_id=chat_data.session_id,
        role="user",
        content=chat_data.message
    )
    await db.flush()

    # RAG 问答
    answer, sources = await rag_service.chat_with_rag(
        query=chat_data.message,
        session_id=chat_data.session_id,
        tenant_id=current_user.tenant_id,
        db=db
    )

    # 保存 AI 回复
    assistant_message = await session_service.add_message(
        db=db,
        session_id=chat_data.session_id,
        role="assistant",
        content=answer,
        sources=sources
    )
    await db.commit()

    return ChatResponse(
        session_id=chat_data.session_id,
        message=answer,
        sources=sources
    )


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    获取会话的聊天历史

    Args:
        session_id: 会话 ID
        current_user: 当前登录用户
        db: 数据库会话

    Returns:
        消息列表
    """
    # 验证会话
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

    # 获取消息
    messages = await session_service.get_session_messages(
        db=db,
        session_id=session_id,
        tenant_id=current_user.tenant_id
    )

    return ApiResponse.success(data=[m.model_dump() for m in messages])