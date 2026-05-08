"""
问答 API 路由模块
提供聊天接口和历史消息查询
"""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from models.schemas import ChatRequest, ChatResponse
from services import session_service, rag_service
from api.auth import get_current_user
from models.schemas import UserResponse
from utils.common import ApiResponse

router = APIRouter(prefix="/chat", tags=["问答"])


@router.post("")
async def chat(
    chat_data: ChatRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """流式问答"""
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

    await session_service.add_message(
        db=db,
        session_id=chat_data.session_id,
        role="user",
        content=chat_data.message
    )
    await db.flush()

    async def event_generator():
        full_answer = ""
        sources = []
        generator = rag_service.chat_with_rag(
            query=chat_data.message,
            session_id=chat_data.session_id,
            tenant_id=current_user.tenant_id,
            db=db
        )
        
        try:
            async for content in generator:
                if content.startswith("event: done\n"):
                    try:
                        json_str = content.replace("event: done\ndata: ", "").replace("\n\n", "")
                        data = json.loads(json_str)
                        sources = data.get("sources", [])
                    except Exception:
                        pass
                    continue

                full_answer += content
                yield f"event: text\ndata: {json.dumps({'content': content})}\n\n"
        except Exception as e:
            logger.error(f"流式生成异常: {e}")
            yield f"event: text\ndata: {json.dumps({'content': '服务异常'})}\n\n"
            return

        await session_service.add_message(
            db=db,
            session_id=chat_data.session_id,
            role="assistant",
            content=full_answer,
            sources=sources
        )
        await db.commit()

        yield f"event: done\ndata: {json.dumps({'sources': sources})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: int,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """获取会话的聊天历史"""
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

    messages = await session_service.get_session_messages(
        db=db,
        session_id=session_id,
        tenant_id=current_user.tenant_id,
        user_id=current_user.id
    )

    return ApiResponse.success(data=[m.model_dump() for m in messages])
