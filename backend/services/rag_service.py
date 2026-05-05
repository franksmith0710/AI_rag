"""
RAG 检索与问答服务模块
负责：1) 混合检索 (向量+BM25) 2) 重排序 3) LLM 答案生成
核心流程：用户问题 → 检索 → 重排序 → 生成答案
"""
import os
import logging
from typing import List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
import jieba

from models.db_models import DocumentChunk, Session as SessionModel, Message
from core.config import get_settings
from core.logging_config import setup_logging

logger = setup_logging("rag_service")
from core.chroma_conn import similarity_search
from utils.rerank import rerank_documents

settings = get_settings()

llm_client = None


def get_llm_client() -> ChatOpenAI:
    """获取 DeepSeek LLM 客户端 (单例, LangChain 封装)"""
    global llm_client
    if llm_client is None:
        llm_client = ChatOpenAI(
            model=settings.deepseek_model,
            openai_api_key=settings.deepseek_api_key,
            openai_api_base=settings.deepseek_base_url,
            temperature=0.7,
            max_tokens=1000
        )
    return llm_client


async def vector_search(query: str, tenant_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    向量检索 (Chroma)

    Args:
        query: 用户查询
        tenant_id: 租户 ID
        top_k: 返回结果数

    Returns:
        检索结果列表
    """
    try:
        results = similarity_search(
            tenant_id=tenant_id,
            query=query,
            k=top_k
        )

        documents = []
        for r in results:
            documents.append({
                "document_id": int(r["metadata"].get("document_id", 0)),
                "chunk_index": int(r["metadata"].get("chunk_index", 0)),
                "text": r["text"],
                "score": 1 - r["score"]  # 距离转相似度
            })

        return documents

    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        return []


async def bm25_search(db: AsyncSession, query: str, tenant_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    BM25 关键词检索
    适合精确匹配文档中的关键词

    Args:
        db: 数据库会话
        query: 用户查询
        tenant_id: 租户 ID
        top_k: 返回结果数

    Returns:
        检索结果列表
    """
    try:
        # 获取该租户的所有文档 ID
        docs_result = await db.execute(
            select(DocumentChunk.document_id)
            .join(DocumentChunk.document)
            .where(DocumentChunk.document.has(tenant_id=tenant_id))
            .distinct()
        )
        document_ids = [row[0] for row in docs_result.all()]

        if not document_ids:
            return []

        # 获取所有分块
        chunks_result = await db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id.in_(document_ids))
        )
        all_chunks = chunks_result.scalars().all()

        if not all_chunks:
            return []

        chunk_list = list(all_chunks)

        # 使用 jieba 分词构建 BM25 索引
        tokenized_corpus = [list(jieba.cut(chunk.text)) for chunk in chunk_list]
        bm25 = BM25Okapi(tokenized_corpus)

        # 查询分词并计算分数
        tokenized_query = list(jieba.cut(query))
        bm25_scores = bm25.get_scores(tokenized_query)

        # 构建结果
        results = []
        for idx, score in enumerate(bm25_scores):
            results.append({
                "document_id": chunk_list[idx].document_id,
                "chunk_index": chunk_list[idx].chunk_index,
                "text": chunk_list[idx].text,
                "score": score
            })

        # 按分数降序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        logger.error(f"BM25 检索失败: {e}")
        return []


async def hybrid_search(query: str, tenant_id: int, db: AsyncSession, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    混合检索：向量检索 + BM25
    结合两种检索方式的结果，综合评分排序

    Args:
        query: 用户查询
        tenant_id: 租户 ID
        db: 数据库会话
        top_k: 返回结果数

    Returns:
        合并排序后的结果列表
    """
    # 并行执行两种检索
    vector_results = await vector_search(query, tenant_id, top_k)
    bm25_results = await bm25_search(db, query, tenant_id, top_k)

    # 合并结果，使用 (document_id, chunk_index) 作为唯一标识
    combined = {}

    # 向量结果权重 60%
    for r in vector_results:
        key = (r["document_id"], r["chunk_index"])
        combined[key] = {
            **r,
            "source": "vector",
            "final_score": r["score"] * 0.6
        }

    # BM25 结果权重 40%
    for r in bm25_results:
        key = (r["document_id"], r["chunk_index"])
        if key in combined:
            combined[key]["final_score"] += r["score"] * 0.4
            if combined[key]["source"] == "vector":
                combined[key]["source"] = "hybrid"  # 两者都有则为混合
        else:
            combined[key] = {
                **r,
                "source": "bm25",
                "final_score": r["score"] * 0.4
            }

    # 按综合分数排序
    sorted_results = sorted(combined.values(), key=lambda x: x["final_score"], reverse=True)
    return sorted_results[:top_k]


async def rerank_results(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    使用 BGE-reranker 进行重排序
    进一步提升检索精度
    """
    if not documents:
        return []

    texts = [doc["text"] for doc in documents]
    reranked = rerank_documents(query, texts, top_k=top_k)

    results = []
    for idx, score in reranked:
        doc = documents[idx]
        results.append({
            **doc,
            "rerank_score": float(score)
        })

    return results


async def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    使用 LLM 生成答案
    将检索到的文档作为上下文，结合对话历史生成答案

    Args:
        query: 用户问题
        context_chunks: 检索到的文档片段
        conversation_history: 对话历史 (最近 10 条)

    Returns:
        (答案文本, 参考来源列表)
    """
    if not context_chunks:
        return "抱歉，我没有找到相关的文档来回答您的问题。", []

    # 构建上下文
    context_text = "\n\n".join([
        f"[文档{i+1}]\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])

    # 构建对话历史
    history_text = ""
    if conversation_history:
        history_text = "对话历史:\n" + "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history[-5:]  # 最近 5 轮
        ])

    # 构建 Prompt
    prompt = f"""你是一个专业的企业知识库问答助手。请根据以下参考文档回答用户的问题。

{history_text}

参考文档:
{context_text}

用户问题: {query}

要求:
1. 只根据参考文档回答，不要编造信息
2. 如果文档中没有相关信息，请明确告知用户
3. 回答要准确、简洁、有条理
4. 如果需要，可以引用文档内容

回答:"""

    try:
        llm = get_llm_client()
        from langchain.schema import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content="你是一个专业的企业知识库问答助手。"),
            HumanMessage(content=prompt)
        ]
        response = await llm.ainvoke(messages)
        answer = response.content

        # 构建参考来源
        sources = []
        for chunk in context_chunks[:3]:
            sources.append({
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "document_id": chunk["document_id"],
                "score": chunk.get("rerank_score", chunk.get("score", 0))
            })

        return answer, sources

    except Exception as e:
        logger.error(f"LLM 生成失败: {e}")
        return f"生成答案时出错: {str(e)}", []


async def chat_with_rag(
    query: str,
    session_id: int,
    tenant_id: int,
    db: AsyncSession
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    RAG 聊天主流程

    1. 获取会话历史
    2. 混合检索 (向量 + BM25)
    3. 重排序
    4. LLM 生成答案
    """
    # 获取会话
    session_result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
    session = session_result.scalar_one_or_none()

    if not session:
        return "会话不存在", []

    # 获取历史消息
    messages_result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    )
    messages = list(messages_result.scalars().all())

    # 构建对话历史
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in messages[-10:]
    ]

    # 检索
    search_results = await hybrid_search(query, tenant_id, db, top_k=10)

    if not search_results:
        return "未找到相关文档，请先上传文档到知识库。", []

    # 重排序
    reranked_results = await rerank_results(query, search_results, top_k=5)

    # 生成答案
    answer, sources = await generate_answer(query, reranked_results, conversation_history)

    return answer, sources