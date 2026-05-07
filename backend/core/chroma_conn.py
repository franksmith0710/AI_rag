"""
Chroma 向量数据库连接模块
负责文档向量的存储和检索
使用 Ollama Embedding
"""
import os
import logging
import httpx
from typing import Optional, List, Dict, Any, Union
from langchain_community.vectorstores import Chroma
from .config import get_settings
from .logging_config import setup_logging
import hashlib

logger = setup_logging("chroma_conn")
settings = get_settings()

_chroma_store: Optional[Chroma] = None
_embedding_model = None
_chroma_stores: Dict[int, Chroma] = {}


def _get_ollama_embedding(text: str) -> List[float]:
    """直接使用 httpx 调用 Ollama API 获取 embedding"""
    ollama_url = settings.ollama_host
    if not ollama_url.startswith("http"):
        ollama_url = f"http://{ollama_url}"
    # 确保使用 IPv4 地址 (Windows)
    if "localhost" in ollama_url:
        ollama_url = ollama_url.replace("localhost", "127.0.0.1")
    
    model = settings.ollama_embed_model or "bge-m3"
    
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": text}
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]


def get_embedding_model():
    """
    获取自定义 Embedding 模型
    使用直接调用方式避免 langchain_ollama 连接问题
    """
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    class CustomOllamaEmbeddings:
        """自定义 Ollama Embedding 包装"""
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            logger.info(f"embed_documents: 接收 {len(texts)} 个文本")
            results = []
            for i, t in enumerate(texts):
                try:
                    emb = _get_ollama_embedding(t)
                    logger.info(f"embed_documents[{i}]: 成功, 向量长度={len(emb)}")
                    results.append(emb)
                except Exception as e:
                    logger.error(f"embed_documents[{i}]: 失败 - {e}")
                    results.append([])
            return results
        
        def embed_query(self, text: str) -> List[float]:
            logger.info(f"embed_query: 文本长度={len(text)}")
            try:
                result = _get_ollama_embedding(text)
                logger.info(f"embed_query: 成功, 向量长度={len(result)}")
                return result
            except Exception as e:
                logger.error(f"embed_query: 失败 - {e}")
                return []
    
    _embedding_model = CustomOllamaEmbeddings()
    logger.info(f"使用 Ollama Embedding: {settings.ollama_embed_model or 'bge-m3'}")
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
    为租户创建独立的向量集合（带缓存）
    每个租户的数据完全隔离

    Args:
        tenant_id: 租户 ID

    Returns:
        该租户的 Chroma 实例
    """
    # 检查缓存
    if tenant_id in _chroma_stores:
        return _chroma_stores[tenant_id]

    # 集合名称格式: tenant_{id}_documents
    collection_name = f"tenant_{tenant_id}_documents"
    persist_dir = os.path.join(settings.chroma_persist_dir, collection_name)
    os.makedirs(persist_dir, exist_ok=True)

    store = Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embedding_model()
    )

    # 缓存
    _chroma_stores[tenant_id] = store
    return store


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