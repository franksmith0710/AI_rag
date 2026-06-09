"""
Chroma 向量数据库连接模块
负责文档向量的存储和检索
使用 ONNX Runtime 加载本地 BGE-M3 模型
"""
import os
import logging
import onnxruntime
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer
from .config import get_settings
import hashlib
import threading

logger = logging.getLogger(__name__)
settings = get_settings()

_chroma_clients: Dict[int, chromadb.PersistentClient] = {}
_client_lock = threading.Lock()
_embedding_lock = threading.Lock()


class LocalEmbedding:
    """本地 Embedding 模型 (BGE-M3 ONNX) — CLS + L2 normalize 已在图内"""
    _session = None
    _tokenizer = None
    _lock = threading.Lock()

    def __init__(self):
        self._load_model()

    def _load_model(self):
        logger.info(f"加载 ONNX Embedding 模型: {settings.embedding_onnx_path}")
        LocalEmbedding._tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model_path)
        LocalEmbedding._session = onnxruntime.InferenceSession(
            settings.embedding_onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        logger.info(f"Embedding ONNX 模型加载完成, providers={LocalEmbedding._session.get_providers()}")

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """嵌入文档（分批处理，节省显存）"""
        logger.info(f"embed_documents: 接收 {len(texts)} 个文本")
        if not texts:
            return []

        all_embeddings = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = LocalEmbedding._tokenizer(
                    batch,
                    return_tensors='np',
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                with LocalEmbedding._lock:
                    outputs = LocalEmbedding._session.run(None, {
                        'input_ids': inputs['input_ids'].astype('int64'),
                        'attention_mask': inputs['attention_mask'].astype('int64'),
                    })[0]
                all_embeddings.extend(outputs.tolist())
            logger.info(f"embed_documents: 成功, 向量数量={len(all_embeddings)}, 向量长度={len(all_embeddings[0]) if all_embeddings else 0}")
            return all_embeddings
        except Exception as e:
            logger.error(f"embed_documents: 批量处理失败 - {e}", exc_info=True)
            return []

    def embed_query(self, text: str) -> List[float]:
        logger.info(f"embed_query: 文本长度={len(text)}")
        try:
            inputs = LocalEmbedding._tokenizer(text, return_tensors='np', max_length=512, truncation=True, padding=True)
            with LocalEmbedding._lock:
                outputs = LocalEmbedding._session.run(None, {
                    'input_ids': inputs['input_ids'].astype('int64'),
                    'attention_mask': inputs['attention_mask'].astype('int64'),
                })[0]
            result = outputs.squeeze().tolist()
            logger.info(f"embed_query: 成功, 向量长度={len(result)}")
            return result
        except Exception as e:
            logger.error(f"embed_query: 失败 - {e}")
            return []


_embedding_model = None


def get_embedding_model() -> LocalEmbedding:
    """获取 Embedding 模型单例（线程安全）"""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                _embedding_model = LocalEmbedding()
    return _embedding_model


def _get_client(tenant_id: int) -> chromadb.PersistentClient:
    """获取租户的 Chroma 客户端（ChromaDB 1.x 兼容）"""
    with _client_lock:
        if tenant_id in _chroma_clients:
            return _chroma_clients[tenant_id]
        persist_dir = os.path.join(settings.chroma_persist_dir, f"tenant_{tenant_id}_documents")
        os.makedirs(persist_dir, exist_ok=True)

        # ChromaDB 1.x 要求 default_tenant 和 default_database 必须存在
        admin = chromadb.AdminClient(
            Settings(is_persistent=True, persist_directory=persist_dir, anonymized_telemetry=False)
        )
        try:
            admin.create_tenant(name="default_tenant")
        except ValueError:
            pass
        try:
            admin.create_database(name="default_database", tenant="default_tenant")
        except ValueError:
            pass

        _chroma_clients[tenant_id] = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_clients[tenant_id]


def add_documents(
    tenant_id: int,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str] = None,
    batch_size: int = 32,
):
    """添加文档到向量库（分批 embedding + upsert，防止 OOM）"""
    client = _get_client(tenant_id)
    collection_name = f"tenant_{tenant_id}_documents"

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,
                "hnsw:search_ef": 50,
                "hnsw:M": 16
            }
        )

    try:
        collection.count()
    except Exception:
        logger.warning(f"Collection {collection_name} 损坏，正在重建...")
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,
                "hnsw:search_ef": 50,
                "hnsw:M": 16
            }
        )

    em = get_embedding_model()
    total = len(texts)
    upserted = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_metadatas = metadatas[start:end]
        batch_ids = [ids[i] if ids else hashlib.md5(texts[i].encode()).hexdigest()
                     for i in range(start, end)]

        batch_embeddings = em.embed_documents(batch_texts)

        valid_ids = []
        valid_texts = []
        valid_metadatas = []
        valid_embeddings = []

        for i, (text, meta, doc_id, emb) in enumerate(
            zip(batch_texts, batch_metadatas, batch_ids, batch_embeddings)
        ):
            if emb and len(emb) > 0:
                valid_ids.append(doc_id)
                valid_texts.append(text)
                valid_metadatas.append(meta)
                valid_embeddings.append(emb)
            else:
                logger.warning(f"跳过无效文本 {start+i}: 向量为空")

        if valid_ids:
            try:
                collection.upsert(
                    ids=valid_ids,
                    embeddings=valid_embeddings,
                    documents=valid_texts,
                    metadatas=valid_metadatas,
                )
                upserted += len(valid_ids)
            except Exception as e:
                logger.error(f"Chroma upsert 失败: {type(e).__name__}: {e}, 正在尝试重建...")
                try:
                    client.delete_collection(name=collection_name)
                except Exception:
                    pass
                collection = client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "hnsw:construction_ef": 100,
                        "hnsw:search_ef": 50,
                        "hnsw:M": 16
                    }
                )
                collection.upsert(
                    ids=valid_ids,
                    embeddings=valid_embeddings,
                    documents=valid_texts,
                    metadatas=valid_metadatas,
                )
                upserted += len(valid_ids)

        logger.info(f"  进度 {upserted}/{total}")

    if upserted == 0:
        raise RuntimeError("所有文本的向量生成失败")

    logger.info(f"成功添加 {upserted}/{total} 个文档到向量库")


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
    except Exception:
        logger.warning(f"向量集合 {collection_name} 不存在，跳过删除")
        return

    try:
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