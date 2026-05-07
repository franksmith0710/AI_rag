"""
问答 API 路由模块
提供聊天接口和历史消息查询
使用 RAG 技术进行问答
支持流式输出
"""
import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
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


@router.post("/stream")
async def chat_stream(
    chat_data: ChatRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    流式问答接口（适配真流式输出）

    使用 Server-Sent Events (SSE) 实现流式输出
    """
    # 验证会话
    session = await session_service.get_session_by_id(
        db=db,
        session_id=chat_data.session_id,
        tenant_id=current_user.tenant_id,
        user_id=current_user.id
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

    async def generate():
        answer_text = ""
        sources_data = []

        # 流式生成回答
        async for chunk in rag_service.stream_chat(
            query=chat_data.message,
            session_id=chat_data.session_id,
            tenant_id=current_user.tenant_id,
            db=db
        ):
            chunk_type = chunk.get("type")

            # 处理所有 chunk 类型
            if chunk_type == "answer_chunk":
                content = chunk.get("content", "")
                answer_text += content
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': content})}\n\n"

            elif chunk_type == "section":
                section_data = chunk.get("data", {})
                yield f"data: {json.dumps({'type': 'section', 'index': chunk.get('index'), 'data': section_data})}\n\n"

            elif chunk_type == "answer_done":
                # 答案完成，保存完整答案
                if chunk.get("full_content"):
                    answer_text = chunk.get("full_content", answer_text)
                yield f"data: {json.dumps({'type': 'answer_done'})}\n\n"

            elif chunk_type == "sources":
                sources_data = chunk.get("data", [])
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

            elif chunk_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'content': chunk.get('content', '')})}\n\n"

        # 保存 AI 回复到数据库
        if answer_text:
            # 尝试解析 answer_text 为 JSON 提取 answer 字段
            try:
                parsed = json.loads(answer_text)
                save_content = parsed.get("answer", answer_text)
            except (json.JSONDecodeError, TypeError):
                save_content = answer_text
            
            await session_service.add_message(
                db=db,
                session_id=chat_data.session_id,
                role="assistant",
                content=save_content,
                sources=sources_data
            )
            await db.commit()

        # 发送完成信号
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
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
        tenant_id=current_user.tenant_id,
        user_id=current_user.id
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
        tenant_id=current_user.tenant_id,
        user_id=current_user.id
    )

    return ApiResponse.success(data=[m.model_dump() for m in messages])