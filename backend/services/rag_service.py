"""
RAG 检索与问答服务模块
负责：1) 混合检索 (向量+BM25) 2) 重排序 3) LLM 答案生成
核心流程：用户问题 → 检索 → 重排序 → 生成答案
支持结构化 JSON 输出
"""
import os
import json
import logging
from typing import List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
import jieba

from models.db_models import DocumentChunk, Session as SessionModel, Message, Document
from core.config import get_settings
from core.logging_config import setup_logging

logger = setup_logging("rag_service")
from core.chroma_conn import similarity_search
from utils.rerank import rerank_documents

# ==================== RAG 检索参数 ====================
# 混合检索候选数量 (给带重叠的正文 chunk 入围机会)
HYBRID_SEARCH_TOP_K = 10
# 重排后最终保留数量 (只留最相关的，严控 Token)
RERANK_TOP_K = 3
# 相关性阈值兜底 (低分无关内容直接跳过 LLM)
MIN_RELEVANCE_SCORE = 0.1

settings = get_settings()

# 设置 LangSmith 环境变量（确保 langchain tracing 正常工作）
import os
if settings.langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
if settings.langchain_project:
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
if settings.langchain_endpoint:
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

_langsmith_enabled = settings.langchain_tracing_v2 and bool(settings.langchain_api_key)

traceable = None
if _langsmith_enabled:
    try:
        from langsmith import traceable
    except ImportError:
        pass

llm_client = None


def get_llm_client() -> ChatOpenAI:
    """获取 DeepSeek LLM 客户端 (单例, LangChain 封装)"""
    global llm_client
    if llm_client is None:
        from langchain_openai import ChatOpenAI
        # 注意: .env 中的 DEEPSEEK_BASE_URL 已包含 /v1，这里不再重复添加
        llm_client = ChatOpenAI(
            model="deepseek-chat",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.7,
            max_tokens=1000,
            http_client=None,
            timeout=60
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
        all_documents = []

        # 检索当前租户 + 全局租户 (tenant_id=0)
        for tid in [tenant_id, 0]:
            try:
                results = similarity_search(
                    tenant_id=tid,
                    query=query,
                    k=top_k
                )
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

        # 按 score 排序，取 top_k
        all_documents.sort(key=lambda x: x["score"], reverse=True)
        return all_documents[:top_k]

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
        # 获取该租户 + 全局租户的所有文档 ID
        docs_result = await db.execute(
            select(DocumentChunk.document_id)
            .join(DocumentChunk.document)
            .where(DocumentChunk.document.has(Document.tenant_id.in_([tenant_id, 0])))
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

    # 临时跳过 Rerank，直接返回原始结果（等模型下载后再启用）
    return documents[:top_k]


async def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]]
) -> Tuple[str, List[Dict[str, Any]]]:
    if not context_chunks:
        return "未找到相关内容。", []

    # 拼接参考资料
    context_text = "\n\n".join([
        f"【参考{i+1}】{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])

    # 超级干净、强制格式的 Prompt（模型必听）
    prompt = f"""请严格按照文档回答，**只输出答案，不要多余解释、不要开场白、不要结尾客套话**。

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
        llm = get_llm_client()
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content="你是简洁、专业、格式严格整齐的企业知识库助手。只输出答案，无多余内容。"),
            HumanMessage(content=prompt)
        ]

        response = await llm.ainvoke(messages)
        answer = response.content.strip()

        # 清理多余空行、多余空格（最终保险）
        import re
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = re.sub(r' +', ' ', answer)

        sources = []
        for chunk in context_chunks[:3]:
            sources.append({
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "document_id": chunk["document_id"],
                "score": chunk.get("rerank_score", chunk.get("score", 0))
            })

        return answer, sources

    except Exception as e:
        logger.error(f"LLM生成失败: {e}")
        return "服务暂时异常，请稍后再试。", []


async def chat_with_rag(
    query: str,
    session_id: int,
    tenant_id: int,
db: AsyncSession
) -> Tuple[str, List[Dict[str, Any]]]:
    # Get session
    session_result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
    session = session_result.scalar_one_or_none()

    if not session:
        return "Session not found", []

    # Get history messages
    messages_result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    )
    messages = list(messages_result.scalars().all())

    # Build conversation history
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in messages[-10:]
    ]

    # Search
    search_results = await hybrid_search(query, tenant_id, db, top_k=HYBRID_SEARCH_TOP_K)

    if not search_results:
        return "No relevant documents found. Please upload documents to the knowledge base first.", []

    # Rerank
    reranked_results = await rerank_results(query, search_results, top_k=RERANK_TOP_K)

    # 相关性阈值判断：低于 MIN_RELEVANCE_SCORE 则跳过 LLM
    if reranked_results and reranked_results[0].get("final_score", 0) < MIN_RELEVANCE_SCORE:
        logger.info(f"检索结果相关性过低，跳过 LLM 调用 (score={reranked_results[0].get('final_score', 0):.3f})")
        return "未找到相关文档", []

    # Generate answer
    answer, sources = await generate_answer(query, reranked_results, conversation_history)

    return answer, sources


async def stream_chat(
    query: str,
    session_id: int,
    tenant_id: int,
    db: AsyncSession
):
    """
    流式问答
    逐块返回回答内容
    """
    logger.info(f"stream_chat: 开始处理 query={query}, session_id={session_id}")
    
    # 获取会话
    session_result = await db.execute(select(SessionModel).where(SessionModel.id == session_id))
    session = session_result.scalar_one_or_none()

    if not session:
        logger.warning(f"stream_chat: 会话不存在 session_id={session_id}")
        yield {"answer": "会话不存在"}
        return

    # 获取历史消息
    messages_result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    )
    messages = list(messages_result.scalars().all())

    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in messages[-10:]
    ]

    # 检索
    logger.info(f"stream_chat: 开始检索 query={query}")
    search_results = await hybrid_search(query, tenant_id, db, top_k=HYBRID_SEARCH_TOP_K)
    logger.info(f"stream_chat: 检索完成, 找到 {len(search_results)} 条结果")

    if not search_results:
        logger.warning("stream_chat: 未找到相关文档")
        # 添加 type 字段，API 层才能识别
        full_data = {
            "answer": "未找到相关文档，请先上传文档到知识库。",
            "sections": [],
            "sources": []
        }
        yield {"type": "answer_chunk", "content": json.dumps(full_data)}
        yield {"type": "sources", "data": []}
        yield {"type": "answer_done", "full_content": json.dumps({"answer": "未找到相关文档", "sections": []})}
        return

    # 重排序
    logger.info(f"stream_chat: 开始重排序")
    reranked_results = await rerank_results(query, search_results, top_k=RERANK_TOP_K)
    logger.info(f"stream_chat: 重排序完成, 保留 {len(reranked_results)} 条结果")

    # 相关性阈值判断：低于 MIN_RELEVANCE_SCORE 则跳过 LLM
    if reranked_results and reranked_results[0].get("final_score", 0) < MIN_RELEVANCE_SCORE:
        logger.info(f"stream_chat: 检索结果相关性过低 (score={reranked_results[0].get('final_score', 0):.3f})，跳过 LLM")
        full_data = {
            "answer": "未找到相关文档",
            "sections": [],
            "sources": []
        }
        yield {"type": "answer_chunk", "content": json.dumps(full_data)}
        yield {"type": "sources", "data": []}
        yield {"type": "answer_done", "full_content": json.dumps({"answer": "未找到相关文档", "sections": []})}
        return

    # 构建上下文
    context_text = "\n\n".join([
        f"[文档{i+1}]\n{chunk['text']}"
        for i, chunk in enumerate(reranked_results)
    ])

    history_text = ""
    if conversation_history:
        history_text = "对话历史:\n" + "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation_history[-5:]
        ])

    prompt = f"""你是一个专业的企业知识库问答助手。请根据以下参考文档回答用户的问题。

{history_text}

参考文档:
{context_text}

用户问题: {query}

请按以下 JSON 格式输出回答（必须是合法的 JSON，不要包含其他内容）:
{{
    "answer": "一句话简短的结论或回答",
    "sections": [
        {{
            "title": "章节标题（如：一、概述/二、详细说明等）",
            "type": "text|list|table",
            "content": "文本内容（type=text 时使用）",
            "items": ["列表项1", "列表项2"]（type=list 时使用），
            "headers": ["列1", "列2", "列3"]（type=table 时使用），
            "rows": [["行1列1", "行1列2", "行1列3"], ["行2列1", "行2列2", "行2列3"]]（type=table 时使用）
        }}
    ]
}}

要求:
1. 只根据参考文档回答，不要编造信息
2. 如果文档中没有相关信息，answer 填写"未找到相关文档"，sections 为空数组
3. sections 至少包含一个章节
4. 如果内容适合用列表呈现，使用 type="list"
5. 如果内容适合用表格呈现，使用 type="table"
6. 输出必须是合法的 JSON，不要有注释

回答（JSON 格式）:"""

    try:
        logger.info(f"stream_chat: 开始调用 LLM")
        llm = get_llm_client()
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content="你是一个专业的企业知识库问答助手。"),
            HumanMessage(content=prompt)
        ]

        # 使用 ainvoke 获取完整响应（JSON 格式）
        logger.info(f"stream_chat: 调用 DeepSeek API, model=deepseek-chat")
        try:
            response = await llm.ainvoke(messages)
            full_text = response.content
        except Exception as api_error:
            logger.error(f"stream_chat: DeepSeek API 调用失败: {api_error}")
            raise
        logger.info(f"stream_chat: LLM 调用完成, 响应长度={len(full_text)}")
        
        # 解析 JSON
        try:
            # 尝试提取 JSON（可能包含在 ```json ... ``` 或 ``` ... ``` 中）
            json_text = full_text.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]
            json_text = json_text.strip()
            
            result = json.loads(json_text)
        except json.JSONDecodeError:
            # 如果解析失败，返回原始文本
            result = {
                "answer": full_text[:100],
                "sections": [
                    {
                        "title": "回答",
                        "type": "text",
                        "content": full_text
                    }
                ]
            }
        
        # 先构建 sources_data
        sources_data = []
        for chunk in reranked_results[:3]:
            sources_data.append({
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "document_id": chunk["document_id"],
                "score": chunk.get("rerank_score", chunk.get("score", 0))
            })
        
        # 返回完整 JSON 数据（用于前端实时显示）
        full_data = {
            "answer": result.get("answer", ""),
            "sections": result.get("sections", []),
            "sources": sources_data
        }
        
        # 先返回 answer_chunk (包含完整 JSON，前端可实时解析)
        logger.info(f"stream_chat: 返回 answer_chunk, content={result.get('answer', '')[:50]}...")
        yield {"type": "answer_chunk", "content": json.dumps(full_data)}
        
        # 再返回 sections (兼容旧代码)
        for i, section in enumerate(result.get("sections", [])):
            logger.info(f"stream_chat: 返回 section {i}")
            yield {"type": "section", "index": i, "data": section}
        
        # 返回 sources
        logger.info(f"stream_chat: 返回 sources, count={len(sources_data)}")
        yield {"type": "sources", "data": sources_data}
        
        # 最后返回 answer_done
        logger.info(f"stream_chat: 返回 answer_done")
        yield {"type": "answer_done", "full_content": json.dumps(result)}

    except Exception as e:
        logger.error(f"stream_chat: 生成答案出错: {e}")
        yield {"type": "error", "content": f"生成答案时出错: {str(e)}"}