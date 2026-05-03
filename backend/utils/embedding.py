"""
BGE 向量嵌入模块
使用 BGE-m3 模型将文本转换为向量
支持批量嵌入和单条查询嵌入
"""
import numpy as np
from typing import List, Union, Optional
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ..core.config import get_settings

settings = get_settings()

_embedding_model = None  # 模型单例


def get_embedding_model(model_path: Optional[str] = None) -> HuggingFaceBgeEmbeddings:
    """
    获取 BGE 嵌入模型
    使用单例模式避免重复加载模型(模型较大，加载耗时)

    Args:
        model_path: 模型路径，默认从配置读取

    Returns:
        BGE 嵌入模型实例
    """
    global _embedding_model
    if _embedding_model is None:
        model_path = model_path or settings.bge_model_path
        _embedding_model = HuggingFaceBgeEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},  # 使用 CPU 推理
            encode_kwargs={'normalize_embeddings': True}  # L2 归一化，使余弦相似度等价于点积
        )
    return _embedding_model


def embed_texts(texts: Union[str, List[str]]) -> np.ndarray:
    """
    批量将文本转换为向量
    用于文档向量化，将文档 chunks 批量嵌入

    Args:
        texts: 文本或文本列表

    Returns:
        numpy.ndarray，形状为 (n, embedding_dim)，n 为文本数量

    示例:
        >>> texts = ["第一段内容", "第二段内容"]
        >>> vectors = embed_texts(texts)
        >>> print(vectors.shape)
        (2, 1024)
    """
    model = get_embedding_model()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.embed_documents(texts)
    return np.array(embeddings)


def embed_query(query: str) -> np.ndarray:
    """
    将用户查询转换为向量
    用于检索阶段，将问题向量化后进行相似度搜索

    Args:
        query: 用户查询文本

    Returns:
        numpy.ndarray，形状为 (embedding_dim,)

    示例:
        >>> query = "如何创建用户？"
        >>> vector = embed_query(query)
        >>> print(vector.shape)
        (1024,)
    """
    model = get_embedding_model()
    embedding = model.embed_query(query)
    return np.array(embedding)


def get_embedding_dim() -> int:
    """
    获取嵌入向量的维度
    用于创建向量数据库时指定维度

    Returns:
        向量维度 (BGE-m3 为 1024)
    """
    model = get_embedding_model()
    test_embedding = model.embed_query("test")
    return len(test_embedding)