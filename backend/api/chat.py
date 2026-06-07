"""
问答 API 路由模块
提供流式聊天接口和历史消息查询
"""
import asyncio
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from starlette.responses import StreamingResponse
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.config import get_settings
from models.schemas import ChatRequest
from services import session_service, rag_service
from api.auth import get_current_user
from models.schemas import UserResponse
from utils.common import ApiResponse

router = APIRouter(prefix="/chat", tags=["问答"])
settings = get_settings()


def _format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def chat(
    chat_data: ChatRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """流式问答（SSE）"""
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

    cached = None
    if settings.redis_enabled:
        cached_data = await session_service.get_cached_session_from_redis(
            chat_data.session_id, current_user.tenant_id
        )
        if cached_data and cached_data.get("version") == (session.message_version or 0):
            cached = cached_data["messages"]

    await session_service.add_message(
        db=db,
        session_id=chat_data.session_id,
        role="user",
        content=chat_data.message
    )
    await db.flush()

    async def event_stream():
        full_answer = ""
        sources = []

        try:
            generator = rag_service.chat_with_rag(
                query=chat_data.message,
                session_id=chat_data.session_id,
                tenant_id=current_user.tenant_id,
                db=db,
                cached_messages=cached,
            )

            async for item in generator:
                item_type = item.get("type")

                if item_type == "text":
                    content = item.get("content", "")
                    full_answer += content
                    yield _format_sse("text", {"content": content})

                elif item_type == "done":
                    sources = item.get("sources", [])
                    yield _format_sse("done", {"sources": sources})

                elif item_type == "error":
                    error_content = item.get("content", "服务异常")
                    full_answer += error_content
                    yield _format_sse("error", {"content": error_content})

        except asyncio.CancelledError:
            logger.warning(f"流式响应取消 session_id={chat_data.session_id}")
        except Exception as e:
            logger.error(f"流式响应异常: {e}", exc_info=True)
            try:
                yield _format_sse("error", {"content": "服务暂时异常，请稍后再试。"})
            except Exception:
                pass
        finally:
            try:
                if full_answer:
                    await session_service.add_message(
                        db=db,
                        session_id=chat_data.session_id,
                        role="assistant",
                        content=full_answer,
                        sources=sources,
                    )

                    if settings.redis_enabled and session.title in (None, "新会话"):
                        try:
                            prompt = f"根据用户问题生成一个简短的会话标题（不超过15个字）：\n{chat_data.message}"
                            llm = ChatOpenAI(
                                model=settings.llm_rewrite_model or "deepseek-chat",
                                api_key=settings.deepseek_api_key,
                                base_url=settings.deepseek_base_url,
                                temperature=0.1,
                                max_tokens=30,
                            )
                            title = (await llm.ainvoke(prompt)).content.strip()
                            if title:
                                session.title = title
                        except Exception as e:
                            logger.warning(f"标题生成失败: {e}")

                    if settings.redis_enabled:
                        try:
                            session.message_version = (session.message_version or 0) + 1
                            messages = await session_service.get_session_messages(
                                db, chat_data.session_id, current_user.tenant_id, current_user.id
                            )
                            await session_service.cache_session_to_redis(
                                chat_data.session_id, current_user.tenant_id,
                                [m.model_dump(mode='json') for m in messages],
                                session.message_version
                            )
                        except Exception as e:
                            logger.warning(f"Redis 缓存更新失败: {e}")

                logger.info(f"流式完成, answer_len={len(full_answer)}, sources_count={len(sources)}")
            except Exception as e:
                logger.error(f"流式后处理失败: {e}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )


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