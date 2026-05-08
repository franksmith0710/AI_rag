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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi
import jieba

from models.db_models import DocumentChunk, Session as SessionModel, Message, Document
from core.config import get_settings
from core.logging_config import setup_logging

logger = setup_logging("rag_service")
from core.chroma_conn import similarity_search
from utils.rerank import rerank_documents

# ==================== RAG 检索参数 ====================
HYBRID_SEARCH_TOP_K = 10
RERANK_TOP_K = 3
MIN_RELEVANCE_SCORE = 0.1

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
            select(DocumentChunk).where(DocumentChunk.document_id.in_(document_ids))
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
                "score": score
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
        key = (r["document_id"], r["chunk_index"])
        rrf_scores[key] = {**r, "source": "vector", "rrf_score": 1 / (RRF_K + rank)}

    for rank, r in enumerate(bm25_results, 1):
        key = (r["document_id"], r["chunk_index"])
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
) -> AsyncGenerator[str, None]:
    """生成答案（流式）"""
    if conversation_history is None:
        conversation_history = []

    if not context_chunks:
        yield "未找到相关内容。"
        return

    context_text = "\n\n".join([
        f"【参考{i+1}】{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])

    history_text = ""
    if conversation_history:
        for msg in conversation_history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

    prompt = f"""{'历史对话：\n' + history_text if history_text else ''}请严格按照文档回答，**只输出答案，不要多余解释、不要开场白、不要结尾客套话**。

【格式强制规则】
- 用 ## 做一级标题
- 用 - 列要点
- 数字、等级、比例用表格
- 段落之间空一行
- 禁止乱加空格
- 禁止重复内容
- 禁止废话

参考资料：
{context_text}

用户问题：{query}

输出答案：
"""

    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.7,
            max_tokens=1000,
            timeout=60
        )
        messages = [
            SystemMessage(content="你是简洁、专业、格式严格整齐的企业知识库助手。只输出答案，无多余内容。"),
            HumanMessage(content=prompt)
        ]

        buffer = ""
        last_yield_time = time.time()
        async for chunk in llm.astream(messages):
            buffer += chunk.content
            now = time.time()
            if len(buffer) >= 60 or (now - last_yield_time) * 1000 >= 100:
                yield buffer
                buffer = ""
                last_yield_time = now
        if buffer:
            yield buffer

        sources = []
        for chunk in context_chunks[:3]:
            sources.append({
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "document_id": chunk["document_id"],
                "score": chunk.get("rerank_score", chunk.get("score", 0))
            })

        yield f"event: done\ndata: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"LLM生成失败: {e}")
        yield "服务暂时异常，请稍后再试。"


async def chat_with_rag(
    query: str,
    session_id: int,
    tenant_id: int,
    db: AsyncSession
) -> AsyncGenerator[str, None]:
    """RAG 问答（流式）"""
    logger.info(f"chat_with_rag: 开始处理 query={query}, session_id={session_id}")

    session_result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
    session = session_result.scalar_one_or_none()

    if not session:
        logger.warning(f"chat_with_rag: 会话不存在 session_id={session_id}")
        yield "会话不存在"
        return

    logger.info(f"chat_with_rag: 开始检索 query={query}")
    search_results = await hybrid_search(query, tenant_id, db, top_k=HYBRID_SEARCH_TOP_K)
    logger.info(f"chat_with_rag: 检索完成, 找到 {len(search_results)} 条结果")

    if not search_results:
        logger.warning("chat_with_rag: 未找到相关文档")
        yield "未找到相关文档，请先上传文档到知识库。"
        return

    logger.info("chat_with_rag: 开始重排序")
    reranked_results = await rerank_results(query, search_results, top_k=RERANK_TOP_K)
    logger.info(f"chat_with_rag: 重排序完成, 保留 {len(reranked_results)} 条结果")

    if reranked_results and reranked_results[0].get("rerank_score", 0) < MIN_RELEVANCE_SCORE:
        logger.info(f"chat_with_rag: 检索结果相关性过低，跳过 LLM (rerank_score={reranked_results[0].get('rerank_score', 0):.4f})")
        yield "未找到相关文档"
        return

    logger.info("chat_with_rag: 开始调用 LLM")

    messages_result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    )
    messages_list = list(messages_result.scalars().all())

    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in messages_list[-10:]
    ]

    generator = generate_answer(query, reranked_results, conversation_history)
    try:
        async for content in generator:
            yield content
    except Exception as e:
        logger.error(f"LLM 流式异常: {e}")
        yield "服务暂时异常，请稍后再试。"
        return

    logger.info("chat_with_rag: 完成")
