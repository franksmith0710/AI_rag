"""
RAG 检索与问答服务模块
负责：1) 混合检索 (向量+BM25) 2) 重排序 3) LLM 答案生成
核心流程：用户问题 → 检索 → 重排序 → 生成答案
"""
import os
import logging
import re
import json
import time
from typing import List, Dict, Any, Tuple, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi
import jieba

from models.db_models import DocumentChunk, Session as SessionModel, Message, Document
from core.config import get_settings
from core.logging_config import setup_logging
from utils.query_rewrite import is_greeting_query, rewrite_query

logger = setup_logging("rag_service")
from core.chroma_conn import similarity_search
from utils.rerank import rerank_documents

# ==================== RAG 检索参数 ====================
HYBRID_SEARCH_TOP_K = 10
RERANK_TOP_K = 3

# LLM 生成参数
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1500
LLM_TIMEOUT = 60

# 流式输出参数
STREAM_BUFFER_MIN_CHARS = 50
STREAM_BUFFER_INTERVAL_MS = 150

# 来源数量限制
MAX_SOURCES_COUNT = 3

REWRITE_HISTORY_TURNS = 5   # 改写：只取用户历史问句（不含assistant）
LLM_HISTORY_TURNS = 5       # LLM生成：最近5轮，保证流畅对话

settings = get_settings()

if settings.langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
if settings.langchain_project:
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
if settings.langchain_endpoint:
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint




async def vector_search(query: str, tenant_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
    """向量检索 (Chroma)"""
    try:
        all_documents = []
        for tid in [tenant_id, 0]:
            try:
                results = similarity_search(tenant_id=tid, query=query, k=top_k)
                for r in results:
                    all_documents.append({
                        "document_id": int(r["metadata"].get("document_id", 0)),
                        "chunk_index": int(r["metadata"].get("chunk_index", 0)),
                        "text": r["text"],
                        "score": 1 - r["score"],
                        "tenant_id": int(r["metadata"].get("tenant_id", tid))
                    })
            except Exception as e:
                logger.warning(f"租户 {tid} 向量检索失败: {e}")
                continue

        all_documents.sort(key=lambda x: x["score"], reverse=True)
        return all_documents[:top_k]

    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        return []


async def bm25_search(db: AsyncSession, query: str, tenant_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
    """BM25 关键词检索"""
    try:
        docs_result = await db.execute(
            select(DocumentChunk.document_id)
            .join(DocumentChunk.document)
            .where(DocumentChunk.document.has(Document.tenant_id.in_([tenant_id, 0])))
            .distinct()
        )
        document_ids = [row[0] for row in docs_result.all()]
        if not document_ids:
            return []

        chunks_result = await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id.in_(document_ids))
            .options(joinedload(DocumentChunk.document))
        )
        all_chunks = chunks_result.scalars().all()
        if not all_chunks:
            return []

        chunk_list = list(all_chunks)
        tokenized_corpus = [list(jieba.cut(chunk.text)) for chunk in chunk_list]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = list(jieba.cut(query))
        bm25_scores = bm25.get_scores(tokenized_query)

        results = []
        for idx, score in enumerate(bm25_scores):
            results.append({
                "document_id": chunk_list[idx].document_id,
                "chunk_index": chunk_list[idx].chunk_index,
                "text": chunk_list[idx].text,
                "score": score,
                "tenant_id": chunk_list[idx].document.tenant_id if chunk_list[idx].document else tenant_id
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        logger.error(f"BM25 检索失败: {e}")
        return []


async def hybrid_search(query: str, tenant_id: int, db: AsyncSession, top_k: int = 10) -> List[Dict[str, Any]]:
    """混合检索：向量检索 + BM25 + RRF"""
    vector_results = await vector_search(query, tenant_id, top_k)
    bm25_results = await bm25_search(db, query, tenant_id, top_k)

    RRF_K = 60
    rrf_scores = {}

    for rank, r in enumerate(vector_results, 1):
        key = (r.get("tenant_id", tenant_id), r["document_id"], r["chunk_index"])
        rrf_scores[key] = {**r, "source": "vector", "rrf_score": 1 / (RRF_K + rank)}

    for rank, r in enumerate(bm25_results, 1):
        key = (r.get("tenant_id", tenant_id), r["document_id"], r["chunk_index"])
        if key in rrf_scores:
            rrf_scores[key]["rrf_score"] += 1 / (RRF_K + rank)
            rrf_scores[key]["source"] = "hybrid"
        else:
            rrf_scores[key] = {**r, "source": "bm25", "rrf_score": 1 / (RRF_K + rank)}

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_results[:top_k]


async def rerank_results(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """使用 BGE-reranker 进行重排序"""
    if not documents:
        return []

    try:
        doc_texts = [d["text"] for d in documents]
        reranked = rerank_documents(query, doc_texts, top_k)

        result = []
        for idx, score in reranked:
            doc = documents[idx].copy()
            doc["rerank_score"] = score
            result.append(doc)

        logger.info(f"Rerank 完成，top_score={reranked[0][1]:.4f}" if reranked else "Rerank 完成，无结果")
        return result

    except Exception as e:
        logger.warning(f"Rerank 失败，使用原始排序: {e}")
        return documents[:top_k]


async def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    生成答案（流式）
    
    Returns:
        AsyncGenerator[dict, None] - yield 字典类型:
        - {"type": "text", "content": "..."} 文本片段
        - {"type": "done", "sources": [...]} 完成标记
        - {"type": "error", "content": "..."} 错误信息
    """
    if conversation_history is None:
        conversation_history = []

    if not context_chunks:
        yield {"type": "error", "content": "未找到相关内容。"}
        yield {"type": "done", "sources": []}
        return

    SYSTEM_PROMPT = """你是专业、严格的企业知识库助手。
只根据参考资料回答，不编造、不扩展、不脑补。
无相关内容时，输出：当前文档未收录该问题
格式要求：## 一级标题，- 列要点，段落之间空一行。"""

    context_text = "\n\n".join([
        f"【参考{i+1}】{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])

    def format_history(history, max_turns=5):
        lines = []
        for msg in history[-max_turns:]:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}：{msg['content']}")
        return "\n".join(lines)

    clean_history = format_history(conversation_history, LLM_HISTORY_TURNS)

    prompt = f"""{SYSTEM_PROMPT}

【历史对话】
{clean_history}

【当前问题】
{query}

【参考资料】
{context_text}

请回答：
"""

    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT
        )
        messages = [
            SystemMessage(content="你是专业、格式严格整齐的企业知识库助手。严格基于参考资料回答，保持内容完整性，不过度精简，不编造内容。"),
            HumanMessage(content=prompt)
        ]

        buffer = ""
        last_yield_time = time.time()
        async for chunk in llm.astream(messages):
            buffer += chunk.content
            now = time.time()
            if len(buffer) >= STREAM_BUFFER_MIN_CHARS or (now - last_yield_time) * 1000 >= STREAM_BUFFER_INTERVAL_MS:
                yield {"type": "text", "content": buffer}
                buffer = ""
                last_yield_time = now
        if buffer:
            yield {"type": "text", "content": buffer}

        sources = []
        for chunk in context_chunks[:MAX_SOURCES_COUNT]:
            sources.append({
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "document_id": chunk["document_id"],
                "score": chunk.get("rerank_score", chunk.get("score", 0))
            })

        yield {"type": "done", "sources": sources}

    except Exception as e:
        logger.error(f"LLM生成失败: {e}")
        yield {"type": "error", "content": "服务暂时异常，请稍后再试。"}
        yield {"type": "done", "sources": []}


async def chat_with_rag(
    query: str,
    session_id: int,
    tenant_id: int,
    db: AsyncSession
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    RAG 问答（流式）

    Returns:
        AsyncGenerator[Dict[str, Any], None] - 透传 generate_answer 的字典类型
    """
    logger.info(f"chat_with_rag: 开始处理 query={query}, session_id={session_id}")

    session_result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
    session = session_result.scalar_one_or_none()

    if not session:
        logger.warning(f"chat_with_rag: 会话不存在 session_id={session_id}")
        yield {"type": "error", "content": "会话不存在"}
        yield {"type": "done", "sources": []}
        return

    if is_greeting_query(query):
        logger.info(f"chat_with_rag: 问候语跳过检索")
        yield {"type": "text", "content": "你好，有什么可以帮您的？"}
        yield {"type": "done", "sources": []}
        return

    messages_result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    )
    messages_list = list(messages_result.scalars().all())

    # ===================== 历史拆分 =====================
    # 给【改写+检索】用：只取最近5条用户问句（剔除assistant）
    history_for_rewrite = (
        [
            {"role": msg.role, "content": msg.content}
            for msg in messages_list
            if msg.role == "user"
        ][-REWRITE_HISTORY_TURNS:]
        if messages_list else []
    )

    # 给【LLM生成】用：取最近5轮原始对话
    history_for_llm = [
        {"role": msg.role, "content": msg.content}
        for msg in messages_list[-LLM_HISTORY_TURNS:]
    ] if messages_list else []

    original_query = query

    # ===================== 执行改写（真正用history） =====================
    rewritten_query = rewrite_query(original_query, history_for_rewrite)
    if rewritten_query != original_query:
        logger.info(f"chat_with_rag: Query 改写 '{original_query}' -> '{rewritten_query}'")
        search_query = rewritten_query
    else:
        logger.info(f"chat_with_rag: Query 无需改写")
        search_query = original_query

    logger.info(f"chat_with_rag: 开始检索 query={search_query}")
    search_results = await hybrid_search(search_query, tenant_id, db, top_k=HYBRID_SEARCH_TOP_K)
    logger.info(f"chat_with_rag: 检索完成, 找到 {len(search_results)} 条结果")

    if not search_results:
        logger.warning("chat_with_rag: 未找到相关文档")
        yield {"type": "error", "content": "未找到相关文档，请先上传文档到知识库。"}
        yield {"type": "done", "sources": []}
        return

    logger.info(f"chat_with_rag: 开始重排 rerank_top_k={RERANK_TOP_K}")
    reranked = await rerank_results(search_query, search_results, RERANK_TOP_K)

    threshold = settings.reranker_threshold
    filtered = [r for r in reranked if r.get("rerank_score", 0) >= threshold]

    if not filtered:
        logger.warning(f"chat_with_rag: 重排结果均低于阈值 {threshold}")
        yield {"type": "error", "content": "当前文档未收录该问题"}
        yield {"type": "done", "sources": []}
        return

    relevant_results = filtered
    logger.info(f"chat_with_rag: 阈值过滤后保留 {len(relevant_results)} 条结果")

    # TODO: 后续替换为四层防御校验
    # 第二层：实体关键词硬匹配
    # 第三层：摘要语义校验
    # 第四层：阈值分数

    logger.info("chat_with_rag: 开始调用 LLM")

    generator = generate_answer(
        query=original_query,
        context_chunks=relevant_results,
        conversation_history=history_for_llm
    )
    try:
        async for item in generator:
            yield item
    except Exception as e:
        logger.error(f"LLM 流式异常: {e}")
        yield {"type": "error", "content": "服务暂时异常，请稍后再试。"}
        yield {"type": "done", "sources": []}
        return

    logger.info("chat_with_rag: 完成")
