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
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
from .config import get_settings
from .logging_config import setup_logging
import hashlib

logger = logging.getLogger(__name__)
settings = get_settings()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_chroma_clients: Dict[int, chromadb.PersistentClient] = {}


class LocalEmbedding:
    """本地 Embedding 模型 (BGE-M3)"""
    _model = None
    _tokenizer = None

    def __init__(self):
        self._load_model()

    def _load_model(self):
        model_path = os.environ.get("EMBEDDING_MODEL_PATH", "/models/BAAI/bge-m3")
        logger.info(f"加载本地 Embedding 模型: {model_path}, 设备: {DEVICE}")

        LocalEmbedding._tokenizer = AutoTokenizer.from_pretrained(model_path)
        LocalEmbedding._model = AutoModel.from_pretrained(model_path).to(DEVICE)
        LocalEmbedding._model.eval()

        logger.info("Embedding 模型加载完成")

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"embed_documents: 接收 {len(texts)} 个文本")
        if not texts:
            return []

        try:
            inputs = LocalEmbedding._tokenizer(
                texts,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = LocalEmbedding._model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                results = embeddings.float().cpu().tolist()

            if len(results) == 1:
                results = [results[0]]
            logger.info(f"embed_documents: 成功, 向量数量={len(results)}, 向量长度={len(results[0]) if results else 0}")
            return results
        except Exception as e:
            logger.error(f"embed_documents: 批量处理失败 - {e}", exc_info=True)
            return []

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


_embedding_model = None


def get_embedding_model() -> LocalEmbedding:
    """获取 Embedding 模型单例"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = LocalEmbedding()
    return _embedding_model


def _get_client(tenant_id: int) -> chromadb.PersistentClient:
    """获取租户的 Chroma 客户端"""
    if tenant_id not in _chroma_clients:
        persist_dir = os.path.join(settings.chroma_persist_dir, f"tenant_{tenant_id}_documents")
        os.makedirs(persist_dir, exist_ok=True)
        _chroma_clients[tenant_id] = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_clients[tenant_id]


def add_documents(
    tenant_id: int,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str] = None
):
    """添加文档到向量库"""
    client = _get_client(tenant_id)
    collection_name = f"tenant_{tenant_id}_documents"

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 200,
                "hnsw:M": 32
            }
        )

    embeddings = get_embedding_model().embed_documents(texts)
    valid_texts = []
    valid_metadatas = []
    valid_ids = []
    valid_embeddings = []

    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        if emb and len(emb) > 0:
            valid_texts.append(text)
            valid_metadatas.append(metadatas[i])
            valid_ids.append(ids[i] if ids else hashlib.md5(text.encode()).hexdigest())
            valid_embeddings.append(emb)
        else:
            logger.warning(f"跳过无效文本 {i}: 向量为空")

    if not valid_texts:
        logger.error("所有文本的向量生成失败")
        raise RuntimeError("所有文本的向量生成失败")

    try:
        collection.upsert(
            ids=valid_ids,
            embeddings=valid_embeddings,
            documents=valid_texts,
            metadatas=valid_metadatas
        )
        logger.info(f"成功添加 {len(valid_texts)} 个文档到向量库")
    except Exception as e:
        logger.error(f"Chroma 添加文档失败: {type(e).__name__}: {e}", exc_info=True)
        raise RuntimeError(f"Chroma 添加文档失败: {type(e).__name__}: {e}")


def similarity_search(
    tenant_id: int,
    query: str,
    k: int = 10,
    filter_metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """相似性搜索"""
    client = _get_client(tenant_id)
    collection_name = f"tenant_{tenant_id}_documents"

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        logger.warning(f"集合 {collection_name} 不存在")
        return []

    query_embedding = get_embedding_model().embed_query(query)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata
        )

        search_results = []
        if results and results.get('documents'):
            for i in range(len(results['documents'][0])):
                search_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                    "distance": results['distances'][0][i] if results.get('distances') else 0,
                    "id": results['ids'][0][i]
                })

        logger.info(f"搜索到 {len(search_results)} 条结果")
        return search_results
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return []


def delete_documents(tenant_id: int, document_ids: List[str] = None, where: Dict[str, Any] = None):
    """删除文档"""
    client = _get_client(tenant_id)
    collection_name = f"tenant_{tenant_id}_documents"

    try:
        collection = client.get_collection(name=collection_name)
        if document_ids:
            collection.delete(ids=document_ids)
        elif where:
            collection.delete(where=where)
        logger.info(f"删除文档成功")
    except Exception as e:
        logger.error(f"删除文档失败: {e}")


def delete_collection(tenant_id: int):
    """删除整个集合"""
    client = _get_client(tenant_id)
    collection_name = f"tenant_{tenant_id}_documents"

    try:
        client.delete_collection(name=collection_name)
        if tenant_id in _chroma_clients:
            del _chroma_clients[tenant_id]
        logger.info(f"删除集合 {collection_name}")
    except Exception as e:
        logger.error(f"删除集合失败: {e}")


def get_collection_stats(tenant_id: int) -> Dict[str, Any]:
    """获取集合统计信息"""
    client = _get_client(tenant_id)
    collection_name = f"tenant_{tenant_id}_documents"

    try:
        collection = client.get_collection(name=collection_name)
        return {
            "name": collection.name,
            "count": collection.count(),
            "tenant_id": tenant_id
        }
    except Exception:
        return {"name": collection_name, "count": 0, "tenant_id": tenant_id}