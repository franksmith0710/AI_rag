"""
Chroma 向量数据库连接模块
负责文档向量的存储和检索
使用本地 BGE-M3 模型进行 Embedding
支持 GPU 加速
"""
import os
import logging
import torch
from typing import Optional, List, Dict, Any
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
from .config import get_settings
from .logging_config import setup_logging
import hashlib

logger = setup_logging("chroma_conn")
settings = get_settings()

_chroma_store: Optional[Chroma] = None
_chroma_stores: Dict[int, Chroma] = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL_PATH = "D:/hf_models/BAAI/bge-m3"


class LocalEmbedding:
    """
    本地 BGE-M3 Embedding 模型
    直接使用 transformers 加载模型，支持 GPU 加速
    """

    _instance = None
    _tokenizer = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        if LocalEmbedding._model is None:
            logger.info(f"加载本地 Embedding 模型: {EMBEDDING_MODEL_PATH}, 设备: {DEVICE}")
            LocalEmbedding._tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
            LocalEmbedding._model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
            if DEVICE == "cuda":
                LocalEmbedding._model = LocalEmbedding._model.half()
            LocalEmbedding._model.to(DEVICE)
            LocalEmbedding._model.eval()
            logger.info("Embedding 模型加载完成")

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"embed_documents: 接收 {len(texts)} 个文本")
        results = []
        for i, t in enumerate(texts):
            try:
                inputs = LocalEmbedding._tokenizer(t, return_tensors='pt', max_length=512, truncation=True, padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = LocalEmbedding._model(**inputs)
                    embedding = self._mean_pooling(outputs, inputs['attention_mask'])
                    emb = embedding.squeeze().float().cpu().tolist()
                results.append(emb)
                logger.info(f"embed_documents[{i}]: 成功, 向量长度={len(emb)}")
            except Exception as e:
                logger.error(f"embed_documents[{i}]: 失败 - {e}")
                results.append([])
        return results

    def embed_query(self, text: str) -> List[float]:
        logger.info(f"embed_query: 文本长度={len(text)}")
        try:
            inputs = LocalEmbedding._tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = LocalEmbedding._model(**inputs)
                embedding = self._mean_pooling(outputs, inputs['attention_mask'])
                result = embedding.squeeze().float().cpu().tolist()
            logger.info(f"embed_query: 成功, 向量长度={len(result)}")
            return result
        except Exception as e:
            logger.error(f"embed_query: 失败 - {e}")
            return []


def get_embedding_model() -> LocalEmbedding:
    """获取 Embedding 模型单例"""
    return LocalEmbedding()


def get_chroma_store() -> Chroma:
    """
    获取默认的 Chroma 存储实例
    用于兼容旧代码，新代码建议使用 create_document_collection
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
    """
    if tenant_id in _chroma_stores:
        return _chroma_stores[tenant_id]

    collection_name = f"tenant_{tenant_id}_documents"
    persist_dir = os.path.join(settings.chroma_persist_dir, collection_name)
    os.makedirs(persist_dir, exist_ok=True)

    store = Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embedding_model()
    )

    _chroma_stores[tenant_id] = store
    return store


def add_documents(
    tenant_id: int,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str] = None
):
    """添加文档到向量库"""
    collection = create_document_collection(tenant_id)

    if ids is None:
        ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]

    collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    collection.persist()


def similarity_search(
    tenant_id: int,
    query: str,
    k: int = 10,
    filter: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """向量相似度搜索"""
    collection = create_document_collection(tenant_id)

    docs = collection.similarity_search_with_score(
        query=query,
        k=k,
        filter=filter
    )

    results = []
    for doc, score in docs:
        results.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })

    return results


def delete_documents(tenant_id: int, document_id: int):
    """删除指定文档的所有 chunks"""
    collection = create_document_collection(tenant_id)

    try:
        collection.delete(where={"document_id": str(document_id)})
        collection.persist()
    except Exception as e:
        logger.error(f"删除文档向量失败: {e}")


def get_embedding_dim() -> int:
    """获取嵌入向量的维度 (BGE-m3 为 1024)"""
    model = get_embedding_model()
    test_embedding = model.embed_query("test")
    return len(test_embedding)