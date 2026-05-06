"""
BGE 重排序模块
使用 BGE-reranker-base 模型对检索结果进行二次排序
提升检索精度，将最相关的文档排在前面
"""
import os
from typing import List, Tuple, Optional
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from core.config import get_settings

settings = get_settings()

_rerank_model = None  # 模型单例


def get_rerank_model(model_path: Optional[str] = None) -> HuggingFaceCrossEncoder:
    """
    获取 BGE 重排序模型
    使用单例模式避免重复加载

    Args:
        model_path: 模型路径，默认从配置读取

    Returns:
        CrossEncoder 模型实例
    """
    global _rerank_model
    if _rerank_model is None:
        # 优先使用本地路径，如果不存在则使用 HuggingFace 模型名称自动下载
        model_path = model_path or settings.rerank_model_path
        if os.path.exists(model_path):
            model_name = model_path
        else:
            # 使用 HuggingFace 模型名称，首次调用会自动下载
            model_name = "BAAI/bge-reranker-base"
        _rerank_model = HuggingFaceCrossEncoder(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
    return _rerank_model


def rerank_documents(
    query: str,
    documents: List[str],
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    对检索结果进行重排序

    工作原理:
        1. 将 query 和每个 doc 拼接成 [query, doc] 对
        2. 一次性传入模型，获取每对的相关性分数
        3. 按分数降序排列，返回 top_k

    Args:
        query: 用户查询
        documents: 待排序的文档列表
        top_k: 返回前 k 个最相关文档

    Returns:
        List[Tuple[索引, 分数]]，按分数降序排列

    示例:
        >>> query = "如何创建用户？"
        >>> docs = ["创建用户的方法...", "用户权限设置...", "文档上传..."]
        >>> results = rerank_documents(query, docs, top_k=2)
        >>> print(results)
        [(1, 0.95), (0, 0.82)]
    """
    if not documents:
        return []

    model = get_rerank_model()

    # 构建 [query, doc] 对列表
    pairs = [[query, doc] for doc in documents]

    # 一次性预测所有对的分数
    scores = model.predict(pairs)

    # 按分数降序排列
    doc_scores = list(enumerate(scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    return doc_scores[:top_k]


def rerank_with_scores(
    query: str,
    documents: List[str],
    scores: List[float]
) -> List[Tuple[int, float]]:
    """
    结合初始分数进行重排序 (预留接口)

    Args:
        query: 用户查询
        documents: 文档列表
        scores: 初始分数 (如向量检索的相似度)

    Returns:
        重排序后的结果
    """
    if not documents:
        return []

    # 暂时只使用 BGE 重排序结果
    reranked = rerank_documents(query, documents, top_k=len(documents))
    return reranked