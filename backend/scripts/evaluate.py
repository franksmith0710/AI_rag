"""
评估脚本 v7
指标：检索相关性 + QA语义相似度 + 答案接地性
纯本地，无需 LLM，不依赖 RAGAS
"""

import asyncio
import json
import math
import sys
import os
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5433")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PASSWORD", "redis123456")
os.environ.setdefault("EMBEDDING_ONNX_PATH", r"D:\hf_models\BAAI\bge-m3-onnx\bge-m3.onnx")
os.environ.setdefault("EMBEDDING_MODEL_PATH", r"D:\hf_models\BAAI\bge-m3")
os.environ.setdefault("RERANKER_ONNX_PATH", r"D:\hf_models\BAAI\bge-reranker-v2-m3-onnx\bge-reranker-v2-m3.onnx")
os.environ.setdefault("RERANKER_MODEL_PATH", r"D:\hf_models\BAAI\bge-reranker-v2-m3")
os.environ.setdefault("CHROMA_PERSIST_DIR", r"D:\School\AI_rag\backend\vector_store\chroma")
os.environ.setdefault("UPLOAD_DIR", r"D:\School\AI_rag\backend\uploads")
os.environ.setdefault("JWT_SECRET_KEY", "your-secret-key-change-in-production")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

from core.config import get_settings
from core.database import async_session_maker, init_db
from core.chroma_conn import get_embedding_model
from sqlalchemy import select
from models.db_models import User

settings = get_settings()

TEST_DATA_JSON = Path(__file__).parent / "test_data.json"
REPORT_PATH = Path(__file__).parent / "eval_report.json"

RERANK_TOP_K = 3
SEARCH_TOP_K = 10


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    hyp_tokens = list(hypothesis)
    ref_tokens = list(reference)
    m, n = len(hyp_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_test_data() -> List[dict]:
    if not TEST_DATA_JSON.exists():
        logger.error(f"未找到测试文件: {TEST_DATA_JSON}")
        logger.error("请先运行 python scripts/generate_eval_data.py 生成测试数据")
        sys.exit(1)
    with open(TEST_DATA_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)
    logger.info(f"加载测试数据: {len(rows)} 条")
    return rows


async def retrieve_chunks(query: str, tenant_id: int, db) -> list[str]:
    from services.rag_service import hybrid_search, _expand_neighbors

    results = await hybrid_search(query, tenant_id, db, top_k=SEARCH_TOP_K)
    if not results:
        return []

    expanded = await _expand_neighbors(results, db)
    expanded.sort(key=lambda r: r.get("rrf_score", 0), reverse=True)
    top = expanded[:RERANK_TOP_K]

    texts = [r["text"] for r in top if r.get("text")]
    return texts


async def find_eval_tenant(db) -> int:
    result = await db.execute(select(User.tenant_id).limit(1))
    row = result.first()
    if row:
        return row[0]
    logger.error("数据库中无用户，请先注册")
    sys.exit(1)


async def main():
    logger.info("初始化数据库...")
    await init_db()

    logger.info("预热 GPU 模型...")
    em = get_embedding_model()
    import jieba
    jieba.initialize()

    test_data = load_test_data()

    async with async_session_maker() as db:
        tenant_id = await find_eval_tenant(db)
        logger.info(f"使用 tenant_id={tenant_id} 进行评估")

        results_detail = []
        skipped = 0

        for i, row in enumerate(test_data):
            question = row["question"]
            ground_truth = row.get("ground_truth", "")

            logger.info(f"[{i+1}/{len(test_data)}] {question[:50]}")
            chunks = await retrieve_chunks(question, tenant_id, db)

            if not chunks or not ground_truth:
                logger.warning(f"  跳过（无检索结果或无答案）")
                skipped += 1
                continue

            # 1. Retrieval relevance
            q_emb = em.embed_query(question)
            chunk_embs = em.embed_documents(chunks)
            scores = [cosine_similarity(q_emb, c_emb) for c_emb in chunk_embs]
            retrieval_max = max(scores) if scores else 0.0
            retrieval_avg = sum(scores) / len(scores) if scores else 0.0

            # 2. QA semantic similarity: cosine(question_emb, ground_truth_emb)
            gt_emb = em.embed_query(ground_truth)
            qa_sim_val = cosine_similarity(q_emb, gt_emb)

            # 3. Answer grounding: ROUGE-L F1 between ground_truth and each chunk, take max
            grounding_vals = [rouge_l_f1(ground_truth, c) for c in chunks]
            grounding_val = max(grounding_vals) if grounding_vals else 0.0

            results_detail.append({
                "question": question,
                "chunks_count": len(chunks),
                "retrieval_relevance_max": round(retrieval_max, 4),
                "retrieval_relevance_avg": round(retrieval_avg, 4),
                "qa_semantic_similarity": round(qa_sim_val, 4),
                "answer_grounding": round(grounding_val, 4),
            })

            logger.info(f"  检索={retrieval_max:.3f} QA语义={qa_sim_val:.3f} 接地={grounding_val:.3f}")

    if not results_detail:
        logger.error("无有效评估数据")
        sys.exit(1)

    logger.info(f"有效数据 {len(results_detail)} 条, 跳过 {skipped} 条")

    avg_ret_max = sum(r["retrieval_relevance_max"] for r in results_detail) / len(results_detail)
    avg_ret_avg = sum(r["retrieval_relevance_avg"] for r in results_detail) / len(results_detail)
    avg_qa = sum(r["qa_semantic_similarity"] for r in results_detail) / len(results_detail)
    avg_gnd = sum(r["answer_grounding"] for r in results_detail) / len(results_detail)

    report = {
        "total": len(results_detail),
        "skipped": skipped,
        "avg_retrieval_relevance_max": round(avg_ret_max, 4),
        "avg_retrieval_relevance_avg": round(avg_ret_avg, 4),
        "avg_qa_semantic_similarity": round(avg_qa, 4),
        "avg_answer_grounding": round(avg_gnd, 4),
        "results": results_detail,
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"  评估报告 ({len(results_detail)} 条, 跳过 {skipped} 条)")
    print("=" * 70)
    print(f"  检索相关性 max:      {avg_ret_max:.4f}")
    print(f"  检索相关性 avg:      {avg_ret_avg:.4f}")
    print(f"  QA 语义相似度:       {avg_qa:.4f}")
    print(f"  答案接地性:          {avg_gnd:.4f}")
    print(f"  报告已保存: {REPORT_PATH}")
    print("=" * 70)

    for r in results_detail:
        flag = "[OK]" if r["retrieval_relevance_max"] >= 0.6 else "[WARN]" if r["retrieval_relevance_max"] >= 0.35 else "[FAIL]"
        q = r['question'][:40]
        try:
            print(f"  {flag} R={r['retrieval_relevance_max']:.2f} Q={r['qa_semantic_similarity']:.2f} G={r['answer_grounding']:.2f} | {q}")
        except UnicodeEncodeError:
            print(f"  {flag} R={r['retrieval_relevance_max']:.2f} Q={r['qa_semantic_similarity']:.2f} G={r['answer_grounding']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
