"""
重排序模块
使用 transformers 直接加载本地 BAAI/bge-reranker-v2-m3 模型
支持 GPU 加速
"""
import logging
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from core.config import get_settings

logger = logging.getLogger("rerank")
settings = get_settings()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_reranker_model = None
_reranker_tokenizer = None


def _get_reranker():
    """获取 reranker 模型单例，延迟加载"""
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        model_path = settings.reranker_model_path
        logger.info(f"加载 reranker 模型: {model_path}, 设备: {DEVICE}")
        _reranker_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if DEVICE == "cuda":
            _reranker_model = _reranker_model.half()
        _reranker_model.to(DEVICE)
        _reranker_model.eval()
        logger.info("Reranker 模型加载完成")
    return _reranker_model, _reranker_tokenizer


def _sigmoid(x: float) -> float:
    return 1 / (1 + pow(2.71828, -x))


def rerank_documents(
    query: str,
    documents: List[str],
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    使用本地 BGE-reranker 模型进行语义重排序

    Args:
        query: 用户查询
        documents: 待排序的文档列表
        top_k: 返回前 k 个最相关文档

    Returns:
        List[Tuple[索引, sigmoid归一化分数]]，按分数降序排列
    """
    if not documents:
        return []

    try:
        model, tokenizer = _get_reranker()
        pairs = [[query, doc] for doc in documents]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            raw_scores = outputs.logits.squeeze(-1).float().cpu().tolist()

        scores = [_sigmoid(s) for s in raw_scores]

        doc_scores = list(enumerate(scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        result = [(idx, score) for idx, score in doc_scores[:top_k]]

        top = result[0][1] if result else 0.0
        logger.info(f"Rerank 完成, top_score={top:.4f}")
        return result

    except Exception as e:
        logger.warning(f"Rerank 失败: {e}，使用原始排序")
        return [(i, 1.0) for i in range(len(documents))]


def rerank_with_scores(
    query: str,
    documents: List[str],
    scores: List[float]
) -> List[Tuple[int, float]]:
    """结合初始分数进行重排序（保留原有接口）"""
    if not documents:
        return []
    doc_scores = list(enumerate(scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:len(documents)]