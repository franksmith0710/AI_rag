"""
RAG 检索与问答服务模块
负责：1) 混合检索 (向量+BM25) 2) 重排序 3) LLM 答案生成
核心流程：用户问题 → 检索 → 重排序 → 生成答案
"""
import asyncio
import os
import logging
import time
from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import joinedload
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi
import jieba

from models.db_models import DocumentChunk, Session as SessionModel, Message, Document
from core.config import get_settings
from core.logging_config import setup_logging
from utils.query_rewrite import is_greeting_query, rewrite_query, expand_query_variants
from services import session_service

logger = setup_logging("rag_service")
from core.chroma_conn import similarity_search
from utils.rerank import rerank_documents

# ==================== RAG 检索参数 ====================
HYBRID_SEARCH_TOP_K = 10
RERANK_TOP_K = 3
NEIGHBOR_COUNT = 1

# LLM 生成参数
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1500
LLM_TIMEOUT = 60

# 流式输出参数
STREAM_BUFFER_MIN_CHARS = 50
STREAM_BUFFER_INTERVAL_MS = 150

# 来源数量限制
MAX_SOURCES_COUNT = 3

REWRITE_HISTORY_TURNS = 4   # 改写：只取用户历史问句（不含assistant，不含当前）
MAX_HISTORY_TOKENS = 4000   # LLM生成：历史对话 token 预算（不含当前 query）

settings = get_settings()

# ── BM25 索引缓存 ──
_bm25_cache: Dict[int, dict] = {}  # tenant_id → {chunks, bm25}
BM25_CACHE_MAX = 5


def invalidate_bm25_cache(tenant_id: int = None):
    """清除 BM25 缓存"""
    global _bm25_cache
    if tenant_id is not None:
        _bm25_cache.pop(tenant_id, None)
    else:
        _bm25_cache.clear()


def _get_or_build_bm25(tenant_id: int, chunks: list):
    """获取或构建 BM25 索引（带缓存）"""
    if tenant_id in _bm25_cache:
        return _bm25_cache[tenant_id]
    # LRU 淘汰
    if len(_bm25_cache) >= BM25_CACHE_MAX:
        oldest_key = next(iter(_bm25_cache))
        del _bm25_cache[oldest_key]
    # jieba 分词
    from utils.splitter import _jieba_cut_for_bm25
    tokenized = [_jieba_cut_for_bm25(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    _bm25_cache[tenant_id] = {"chunks": chunks, "bm25": bm25, "tokenized": tokenized}
    return _bm25_cache[tenant_id]


if settings.langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
if settings.langchain_project:
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
if settings.langchain_endpoint:
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint



_token_counter = None


def count_tokens(text: str) -> int:
    global _token_counter
    if _token_counter is None:
        import tiktoken
        _token_counter = tiktoken.get_encoding("cl100k_base")
    return len(_token_counter.encode(text))


def build_token_window(
    messages: List[Dict[str, str]], max_tokens: int
) -> List[Dict[str, str]]:
    """从后往前累加 token，超出预算截断"""
    total = 0
    window = []
    for msg in reversed(messages):
        t = count_tokens(msg["content"])
        if total + t > max_tokens:
            break
        window.insert(0, msg)
        total += t
    return window


async def summarize_history(cut_messages: List[Dict[str, str]]) -> str:
    """将被裁掉的早期对话压缩为摘要"""
    text = "\n".join(f"{m['role']}: {m['content']}" for m in cut_messages)
    prompt = f"""以下是对话早期的部分内容，请用2-3句话概括核心话题和关键信息：

{text}

概括："""
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.1,
            max_tokens=256,
        )
        return (await llm.ainvoke(prompt)).content.strip()
    except Exception as e:
        logger.warning(f"历史摘要生成失败: {e}")
        return ""


async def vector_search(query: str, tenant_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
    """向量检索 (Chroma)"""
    logger.info(f"向量检索: tenant={tenant_id}, query='{query[:50]}...'")
    try:
        all_documents = []
        for tid in [tenant_id, 0]:
            try:
                def _vector_search_sync(tid=tid):
                    return similarity_search(tenant_id=tid, query=query, k=top_k)
                results = await asyncio.to_thread(_vector_search_sync)
                for r in results:
                    all_documents.append({
                        "document_id": int(r["metadata"].get("document_id", 0)),
                        "chunk_index": int(r["metadata"].get("chunk_index", 0)),
                        "text": r.get("document", r.get("text", "")),
                        "score": 1 - r["distance"],
                        "tenant_id": int(r["metadata"].get("tenant_id", tid))
                    })
            except Exception as e:
                logger.warning(f"租户 {tid} 向量检索失败: {e}")
                continue

        all_documents.sort(key=lambda x: x["score"], reverse=True)
        result = all_documents[:top_k]
        logger.info(f"向量检索完成: {len(result)} 条结果")
        return result

    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        return []


async def bm25_search(db: AsyncSession, query: str, tenant_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
    """BM25 关键词检索（带缓存）"""
    logger.info(f"BM25 检索: tenant={tenant_id}, query='{query[:50]}...'")
    try:
        # 检查缓存
        cache_entry = _bm25_cache.get(tenant_id)
        if cache_entry is None:
            # 缓存未命中，查询数据库构建索引
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
                select(DocumentChunk, DocumentChunk.document_id)
                .join(DocumentChunk.document)
                .where(DocumentChunk.document_id.in_(document_ids))
                .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
            )
            chunks = []
            for row in chunks_result.all():
                chunk = row[0]
                chunks.append({
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "source": f"document_{chunk.document_id}",
                    "score": 0
                })

            if not chunks:
                return []

            cache_entry = await asyncio.to_thread(_get_or_build_bm25, tenant_id, chunks)

        # 使用缓存的 BM25 索引评分
        bm25 = cache_entry["bm25"]
        chunks = cache_entry["chunks"]

        def _score_sync():
            from utils.splitter import _jieba_cut_for_bm25
            import numpy as np
            query_tokens = _jieba_cut_for_bm25(query)
            return bm25.get_scores(query_tokens)

        scores = await asyncio.to_thread(_score_sync)
        import numpy as np

        results = []
        for idx in np.argsort(scores)[::-1][:top_k]:
            if scores[idx] > 0:
                chunk = chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                results.append(chunk)

        return results

    except Exception as e:
        logger.error(f"BM25 检索失败: {str(e)}")
        return []


async def hybrid_search(query: str, tenant_id: int, db: AsyncSession, top_k: int = 10) -> List[Dict[str, Any]]:
    """混合检索：向量检索 + BM25 + RRF"""
    logger.info(f"混合检索: tenant={tenant_id}")
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
    result = sorted_results[:top_k]
    logger.info(f"RRF 融合完成: vector={len(vector_results)} + BM25={len(bm25_results)} → rrf={len(result)}")
    return result


async def _expand_neighbors(
    chunks: List[Dict[str, Any]],
    db: AsyncSession,
    neighbor_count: int = NEIGHBOR_COUNT,
) -> List[Dict[str, Any]]:
    """邻居扩展：为每条结果补 ±N 个邻居，不合并，保持独立 chunk"""
    if not chunks or neighbor_count < 1:
        return chunks

    seen = set()
    conditions = []
    for r in chunks:
        key = (r["document_id"], r["chunk_index"])
        if key in seen:
            continue
        seen.add(key)
        for offset in range(-neighbor_count, neighbor_count + 1):
            n_idx = r["chunk_index"] + offset
            if n_idx >= 0:
                conditions.append(
                    and_(
                        DocumentChunk.document_id == r["document_id"],
                        DocumentChunk.chunk_index == n_idx,
                    )
                )

    if not conditions:
        return chunks

    result = await db.execute(
        select(DocumentChunk)
        .where(or_(*conditions))
        .options(joinedload(DocumentChunk.document))
    )
    rows = result.scalars().all()

    seen_ids = set()
    expanded = []

    for r in chunks:
        key = (r["document_id"], r["chunk_index"])
        if key not in seen_ids:
            seen_ids.add(key)
            expanded.append(r)

    for row in rows:
        key = (row.document_id, row.chunk_index)
        if key not in seen_ids:
            seen_ids.add(key)
            expanded.append({
                "document_id": row.document_id,
                "chunk_index": row.chunk_index,
                "text": row.text,
                "tenant_id": row.document.tenant_id if row.document else 0,
                "score": 0,
            })

    logger.info(f"邻居扩展: {len(chunks)} 条 → {len(expanded)} 条")
    return expanded


async def rerank_results(query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """使用 BGE-reranker 进行重排序"""
    if not documents:
        return []

    try:
        doc_texts = [d["text"] for d in documents]
        reranked = await asyncio.to_thread(rerank_documents, query, doc_texts, top_k)

        result = []
        for idx, score in reranked:
            doc = documents[idx].copy()
            doc["rerank_score"] = score
            result.append(doc)

        return result

    except Exception as e:
        logger.warning(f"Rerank 失败，使用原始排序: {e}")
        return documents[:top_k]


async def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None,
    history_summary: str = "",
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

    def format_history(history):
        lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}：{msg['content']}")
        return "\n".join(lines)

    clean_history = format_history(conversation_history)

    summary_block = f"\n【历史摘要】\n{history_summary}\n" if history_summary else ""

    prompt = f"""{SYSTEM_PROMPT}
{summary_block}
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

        logger.info(f"LLM 调用: model=deepseek-chat, query_len={len(query)}, context_chunks={len(context_chunks)}, history_len={len(conversation_history)}")
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
                "text": chunk["text"],
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
    db: AsyncSession,
    cached_messages: Optional[List[Dict[str, str]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    RAG 问答

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

    # ===================== 加载消息（缓存优先） =====================
    if cached_messages is not None:
        raw_dicts = cached_messages + [{"role": "user", "content": query}]
    else:
        messages_result = await db.execute(
            select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
        )
        raw_orm = list(messages_result.scalars().all())
        raw_dicts = [{"role": m.role, "content": m.content} for m in raw_orm]

    # ===================== 历史拆分 =====================
    # 给【改写+检索】用：只取最近4条用户问句
    history_for_rewrite = (
        [m for m in raw_dicts if m["role"] == "user"][-REWRITE_HISTORY_TURNS:]
        if raw_dicts else []
    )

    # 给【LLM生成】用：token 滑动窗口（排除当前消息）
    recent = raw_dicts[:-1]
    history_for_llm = build_token_window(recent, MAX_HISTORY_TOKENS)

    # 长会话摘要：被窗口裁掉的部分（缓存优先）
    history_summary = ""
    if len(recent) > len(history_for_llm):
        cut_count = len(recent) - len(history_for_llm)
        if cut_count >= 4:
            cached = await session_service.get_cached_summary_from_redis(
                session_id, tenant_id, cut_count
            )
            if cached is not None:
                history_summary = cached
            else:
                cut_messages = recent[:cut_count]
                history_summary = await summarize_history(cut_messages)
                await session_service.cache_summary_to_redis(
                    session_id, tenant_id, cut_count, history_summary
                )

    original_query = query

    # ===================== 执行改写（真正用history） =====================
    rewritten_query = await rewrite_query(original_query, history_for_rewrite)
    if rewritten_query != original_query:
        logger.info(f"chat_with_rag: Query 改写 '{original_query}' -> '{rewritten_query}'")
    else:
        logger.info(f"chat_with_rag: Query 无需改写")

    # ===================== 多语义变体扩展 & 多路检索 =====================
    if settings.query_variant_enabled and settings.query_variant_count > 1:
        variants = await expand_query_variants(rewritten_query, history_for_rewrite, settings.query_variant_count)
        logger.info(f"chat_with_rag: 生成 {len(variants)} 个检索变体: {variants}")
        per_variant_top_k = HYBRID_SEARCH_TOP_K // len(variants) + 2
        tasks = [hybrid_search(v, tenant_id, db, top_k=per_variant_top_k) for v in variants]
        all_results_lists = await asyncio.gather(*tasks)

        seen = {}
        for results in all_results_lists:
            for r in results:
                key = (r["tenant_id"], r["document_id"], r["chunk_index"])
                if key not in seen or r["rrf_score"] > seen[key]["rrf_score"]:
                    seen[key] = r
        merged = sorted(seen.values(), key=lambda x: x["rrf_score"], reverse=True)
        search_results = merged[:HYBRID_SEARCH_TOP_K * 2]
        search_query = rewritten_query
        logger.info(f"chat_with_rag: 多路检索完成, 合并后 {len(search_results)} 条结果")
    else:
        logger.info(f"chat_with_rag: 开始检索 query={rewritten_query}")
        search_results = await hybrid_search(rewritten_query, tenant_id, db, top_k=HYBRID_SEARCH_TOP_K)
        search_query = rewritten_query
        logger.info(f"chat_with_rag: 检索完成, 找到 {len(search_results)} 条结果")

    # ===================== 邻居扩展 =====================
    search_results = await _expand_neighbors(search_results, db)
    search_results = search_results[:15]

    if not search_results:
        logger.warning("chat_with_rag: 未找到相关文档")
        yield {"type": "error", "content": "未找到相关文档，请先上传文档到知识库。"}
        yield {"type": "done", "sources": []}
        return

    logger.info(f"chat_with_rag: 开始重排 rerank_top_k={RERANK_TOP_K}")
    reranked = await rerank_results(search_query, search_results, RERANK_TOP_K)

    relevant_results = [r for r in reranked[:RERANK_TOP_K] if r.get("rerank_score", 0) >= 0.1]

    if not relevant_results:
        logger.warning(f"重排结果均低于阈值 0.1, top_score={reranked[0].get('rerank_score', 0):.4f}" if reranked else "重排无结果")
        yield {"type": "error", "content": "当前文档未收录该问题"}
        yield {"type": "done", "sources": []}
        return

    logger.info("chat_with_rag: 开始调用 LLM")

    generator = generate_answer(
        query=original_query,
        context_chunks=relevant_results,
        conversation_history=history_for_llm,
        history_summary=history_summary,
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
