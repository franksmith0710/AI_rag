"""
Chroma 向量数据库连接模块
负责文档向量的存储和检索
Chroma 是一个轻量级的本地向量数据库，适合开发环境
"""
import os
import logging
from typing import Optional, List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from .config import get_settings
from .logging_config import setup_logging
import hashlib

logger = setup_logging("chroma_conn")
settings = get_settings()

_chroma_store: Optional[Chroma] = None  # Chroma 客户端单例
_embedding_model = None  # BGE 嵌入模型单例


def get_embedding_model():
    """
    获取 Ollama Embedding 模型
    使用本地 Ollama 服务

    Returns:
        OllamaEmbeddings 模型实例
    """
    global _embedding_model
    if _embedding_model is None:
        # 优先使用配置，如果没有则使用本地默认
        base_url = settings.ollama_host if settings.ollama_host else "http://127.0.0.1:11434"
        _embedding_model = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=base_url
        )
    return _embedding_model


def get_chroma_store() -> Chroma:
    """
    获取默认的 Chroma 存储实例
    用于兼容旧代码，新代码建议使用 create_document_collection

    Returns:
        Chroma 实例
    """
    global _chroma_store

    if _chroma_store is None:
        persist_dir = settings.chroma_persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        _chroma_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=get_embedding_model()
        )

    return _chroma_store


def create_document_collection(tenant_id: int) -> Chroma:
    """
    为租户创建独立的向量集合
    每个租户的数据完全隔离

    Args:
        tenant_id: 租户 ID

    Returns:
        该租户的 Chroma 实例
    """
    # 集合名称格式: tenant_{id}_documents
    collection_name = f"tenant_{tenant_id}_documents"
    persist_dir = os.path.join(settings.chroma_persist_dir, collection_name)
    os.makedirs(persist_dir, exist_ok=True)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embedding_model()
    )


def add_documents(
    tenant_id: int,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str] = None
):
    """
    添加文档到向量库

    Args:
        tenant_id: 租户 ID
        texts: 文档文本列表
        metadatas: 元数据列表 (如 document_id, chunk_index 等)
        ids: 可选的文档 ID 列表，默认使用 MD5 哈希
    """
    collection = create_document_collection(tenant_id)

    if ids is None:
        # 使用文本 MD5 哈希作为 ID
        ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]

    # 添加文本到向量库
    collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    collection.persist()  # 持久化保存


def similarity_search(
    tenant_id: int,
    query: str,
    k: int = 10,
    filter: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    向量相似度搜索

    Args:
        tenant_id: 租户 ID
        query: 查询文本
        k: 返回结果数量
        filter: 可选的过滤条件

    Returns:
        搜索结果列表，每项包含 text, metadata, score
    """
    collection = create_document_collection(tenant_id)

    # similarity_search_with_score 返回 (Document, score) 元组列表
    docs = collection.similarity_search_with_score(
        query=query,
        k=k,
        filter=filter
    )

    results = []
    for doc, score in docs:
        results.append({
            "text": doc.page_content,  # 文档内容
            "metadata": doc.metadata,  # 元数据
            "score": score  # 相似度分数 (Chroma 返回的是距离，需转换)
        })

    return results


def delete_documents(tenant_id: int, document_id: int):
    """
    删除指定文档的所有 chunks

    Args:
        tenant_id: 租户 ID
        document_id: 文档 ID
    """
    collection = create_document_collection(tenant_id)

    try:
        # 根据 document_id 过滤删除
        collection.delete(where={"document_id": str(document_id)})
        collection.persist()
    except Exception as e:
        logger.error(f"删除文档向量失败: {e}")


def get_embedding_dim() -> int:
    """
    获取嵌入向量的维度
    用于初始化向量数据库字段

    Returns:
        向量维度 (BGE-m3 为 1024)
    """
    model = get_embedding_model()
    test_embedding = model.embed_query("test")
    return len(test_embedding)