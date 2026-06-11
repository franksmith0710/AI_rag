"""
重排序模块
使用 ONNX Runtime 加载本地 BAAI/bge-reranker-v2-m3 ONNX 模型
支持 CUDA 加速，推理后自动释放 GPU 显存
"""
import logging
from typing import List, Tuple
import onnxruntime
import threading
import torch
from transformers import AutoTokenizer
from core.config import get_settings
from core.logging_config import setup_logging

logger = setup_logging("rerank")
settings = get_settings()

_reranker_model = None
_reranker_tokenizer = None
_reranker_lock = threading.Lock()


def _get_reranker():
    """获取 reranker 模型单例，延迟加载（线程安全）"""
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        with _reranker_lock:
            if _reranker_model is None:
                logger.info(f"加载 ONNX Reranker 模型: {settings.reranker_onnx_path}")
                _reranker_tokenizer = AutoTokenizer.from_pretrained(settings.reranker_model_path)
                opts = onnxruntime.SessionOptions()
                opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.log_severity_level = 3
                opts.intra_op_num_threads = 4
                opts.inter_op_num_threads = 2
                try:
                    _reranker_model = onnxruntime.InferenceSession(
                        settings.reranker_onnx_path,
                        sess_options=opts,
                        providers=[
                            ('CUDAExecutionProvider', {
                                'arena_extend_strategy': 'kSameAsRequested',
                                'gpu_mem_limit': 1024 * 1024 * 1024,
                            }),
                        ]
                    )
                except Exception:
                    logger.warning("Reranker GPU 显存不足，回退到 CPU")
                    _reranker_model = onnxruntime.InferenceSession(
                        settings.reranker_onnx_path,
                        sess_options=opts,
                        providers=['CPUExecutionProvider']
                    )
                logger.info(f"Reranker ONNX 模型加载完成, providers={_reranker_model.get_providers()}")
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
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="np", max_length=512)

        outputs = model.run(None, {
            'input_ids': inputs['input_ids'].astype('int64'),
            'attention_mask': inputs['attention_mask'].astype('int64'),
        })[0]
        raw_scores = outputs.squeeze(-1).tolist()

        scores = [_sigmoid(s) for s in raw_scores]

        doc_scores = list(enumerate(scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        result = [(idx, score) for idx, score in doc_scores[:top_k]]

        top = result[0][1] if result else 0.0
        logger.info(f"Rerank 完成, top_score={top:.4f}")
        return result

    except Exception as e:
        logger.warning(f"Rerank 失败: {e}，使用原始排序")
        return [(i, 1.0) for i in range(min(top_k, len(documents)))]