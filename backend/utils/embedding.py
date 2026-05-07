"""
Ollama 向量嵌入模块
使用 Ollama 的 Embedding 模型将文本转换为向量
支持批量嵌入和单条查询嵌入
"""
import numpy as np
from typing import List, Union, Optional
from langchain_ollama import OllamaEmbeddings
from ..core.config import get_settings

settings = get_settings()

_embedding_model = None  # 模型单例


def get_embedding_model(model_name: Optional[str] = None) -> OllamaEmbeddings:
    """
    获取 Ollama 嵌入模型
    使用单例模式避免重复创建模型实例

    Args:
        model_name: 模型名称，默认从配置读取

    Returns:
        Ollama 嵌入模型实例
    """
    global _embedding_model
    if _embedding_model is None:
        model_name = model_name or settings.ollama_embed_model or "bge-m3"
        ollama_base_url = settings.ollama_host or "http://127.0.0.1:11434"
        _embedding_model = OllamaEmbeddings(
            model=model_name,
            base_url=ollama_base_url
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
    """
    model = get_embedding_model()
    embedding = model.embed_query(query)
    return np.array(embedding)


def get_embedding_dim() -> int:
    """
    获取嵌入向量的维度
    用于创建向量数据库时指定维度

    Returns:
        向量维度 (bge-m3 为 1024)
    """
    model = get_embedding_model()
    test_embedding = model.embed_query("test")
    return len(test_embedding)