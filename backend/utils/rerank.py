"""
重排序模块
使用本地 Ollama rerank 模型 (qllama/bge-reranker-v2-m3) 进行重排序
"""
import httpx
import numpy as np
import logging
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("rerank")

_reranker_model_name = "qllama/bge-reranker-v2-m3"


def _get_embedding(text: str) -> List[float]:
    """使用本地 Ollama 获取 embedding"""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "http://127.0.0.1:11434/api/embed",
            json={"model": _reranker_model_name, "input": text}
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]


def rerank_documents(
    query: str,
    documents: List[str],
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    使用本地 Ollama rerank 模型进行重排序
    通过 embedding 相似度计算进行 rerank

    Args:
        query: 用户查询
        documents: 待排序的文档列表
        top_k: 返回前 k 个最相关文档

    Returns:
        List[Tuple[索引, 分数]]，按相似度降序排列
    """
    if not documents:
        return []

    try:
        q_emb = _get_embedding(query)
        
        scores = []
        for doc in documents:
            d_emb = _get_embedding(doc)
            
            q = np.array(q_emb)
            d = np.array(d_emb)
            dot = float(np.dot(q, d))
            norm_q = float(np.linalg.norm(q))
            norm_d = float(np.linalg.norm(d))
            score = dot / (norm_q * norm_d) if norm_q > 0 and norm_d > 0 else 0.0
            scores.append(score)
        
        doc_scores = list(enumerate(scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Rerank ({_reranker_model_name}): top score = {doc_scores[0][1]:.4f}")
        return doc_scores[:top_k]
    
    except Exception as e:
        logger.warning(f"Rerank failed, using original order: {e}")
        doc_scores = list(enumerate([1.0] * len(documents)))
        return doc_scores[:top_k]


def rerank_with_scores(
    query: str,
    documents: List[str],
    scores: List[float]
) -> List[Tuple[int, float]]:
    """
    结合初始分数进行重排序
    按初始分数排序

    Args:
        query: 用户查询
        documents: 文档列表
        scores: 初始分数 (如向量检索的相似度)

    Returns:
        重排序后的结果
    """
    if not documents:
        return []

    doc_scores = list(enumerate(scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    return doc_scores[:len(documents)]