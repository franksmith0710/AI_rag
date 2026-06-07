"""
轻量评估脚本
对测试 QA 集运行 RAG pipeline, 计算检索命中率 + 语义相似度 + 关键词召回
"""

import asyncio
import csv
import json
import sys
import math
from pathlib import Path
from typing import List, Tuple
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5433")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PASSWORD", "redis123456")
os.environ.setdefault("EMBEDDING_ONNX_PATH", r"D:\hf_models\BAAI\bge-m3-onnx\bge-m3.onnx")
os.environ.setdefault("EMBEDDING_MODEL_PATH", r"D:\hf_models\BAAI\bge-m3")
os.environ.setdefault("RERANKER_ONNX_PATH", r"D:\hf_models\BAAI\bge-reranker-v2-m3-onnx\bge-reranker-v2-m3.onnx")
os.environ.setdefault("RERANKER_MODEL_PATH", r"D:\hf_models\BAAI\bge-reranker-v2-m3")
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-v4-flash")
os.environ.setdefault("CHROMA_PERSIST_DIR", r"D:\School\AI_rag\backend\vector_store\chroma")
os.environ.setdefault("UPLOAD_DIR", r"D:\School\AI_rag\backend\uploads")
os.environ.setdefault("JWT_SECRET_KEY", "your-secret-key-change-in-production")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

from core.config import get_settings
from core.database import async_session_maker, init_db
from core.chroma_conn import get_embedding_model, get_collection_stats
from sqlalchemy import select
from models.db_models import Document, DocumentChunk, Session, User

settings = get_settings()

TEST_DATA = Path(__file__).parent / "test_data.csv"
TEST_DATA_JSON = Path(__file__).parent / "test_data.json"
REPORT_PATH = Path(__file__).parent / "eval_report.json"

def load_test_data() -> List[dict]:
    json_path = Path(__file__).parent / "test_data.json"
    csv_path = Path(__file__).parent / "test_data.csv"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        logger.info(f"加载 JSON 测试数据: {len(rows)} 条")
        return rows

    if csv_path.exists():
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("question", "").strip():
                    rows.append(row)
        logger.info(f"加载 CSV 测试数据: {len(rows)} 条")
        return rows

    logger.error(f"未找到测试文件: {json_path} 或 {csv_path}")
    sys.exit(1)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

async def run_rag(query: str, tenant_id: int, db) -> Tuple[str, List[str]]:
    from services.rag_service import chat_with_rag

    user_result = await db.execute(
        select(User).where(User.tenant_id == tenant_id).limit(1)
    )
    user = user_result.scalar_one_or_none()
    if not user:
        logger.error(f"tenant={tenant_id} 下无用户，请先注册")
        return "（无用户）", []

    session_result = await db.execute(
        select(Session).where(Session.tenant_id == tenant_id).order_by(Session.id.desc()).limit(1)
    )
    session = session_result.scalar_one_or_none()
    if not session:
        session = Session(tenant_id=tenant_id, user_id=user.id, title="eval")
        db.add(session)
        await db.flush()

    answer = ""
    chunks = []
    async for item in chat_with_rag(
        query=query,
        session_id=session.id,
        tenant_id=tenant_id,
        db=db,
    ):
        if item.get("type") == "text":
            answer += item.get("content", "")
        elif item.get("type") == "done":
            chunks = [s.get("text", "") for s in item.get("sources", [])]
        elif item.get("type") == "error":
            answer = item.get("content", "")

    return answer, chunks

def compute_keyword_recall(answer: str, ground_truth: str) -> float:
    import jieba
    if not ground_truth.strip():
        return 0.0
    gt_words = set(jieba.lcut(ground_truth))
    ans_words = set(jieba.lcut(answer))
    if not gt_words:
        return 0.0
    return len(gt_words & ans_words) / len(gt_words)

async def main():
    logger.info("初始化数据库...")
    await init_db()

    logger.info("预热 GPU 模型...")
    em = get_embedding_model()
    from utils.rerank import _get_reranker
    _get_reranker()
    import jieba
    jieba.initialize()

    test_data = load_test_data()

    results = []

    async with async_session_maker() as db:
        for i, row in enumerate(test_data):
            question = row["question"]
            ground_truth = row.get("ground_truth", "")

            logger.info(f"[{i+1}/{len(test_data)}] {question}")
            answer, chunks = await run_rag(question, 0, db)

            gt_embedding = em.embed_query(ground_truth)
            ans_embedding = em.embed_query(answer)
            sim = cosine_similarity(gt_embedding, ans_embedding) if gt_embedding and ans_embedding else 0.0

            kw_recall = compute_keyword_recall(answer, ground_truth)

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "answer_len": len(answer),
                "chunks_count": len(chunks),
                "similarity": round(sim, 4),
                "keyword_recall": round(kw_recall, 4),
            })

            logger.info(f"  similarity={sim:.4f}, keyword_recall={kw_recall:.4f}, answer_len={len(answer)}")

    avg_sim = sum(r["similarity"] for r in results) / len(results) if results else 0
    avg_kr = sum(r["keyword_recall"] for r in results) / len(results) if results else 0

    report = {
        "total": len(results),
        "avg_similarity": round(avg_sim, 4),
        "avg_keyword_recall": round(avg_kr, 4),
        "results": results,
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"  评估报告 ({len(results)} 条)")
    print("=" * 60)
    print(f"  平均语义相似度:   {avg_sim:.4f}")
    print(f"  平均关键词召回:   {avg_kr:.4f}")
    print(f"  报告已保存:       {REPORT_PATH}")
    print("=" * 60)

    for r in results:
        flag = "✅" if r["similarity"] >= 0.7 else "⚠️" if r["similarity"] >= 0.4 else "❌"
        print(f"  {flag} sim={r['similarity']:.4f} kr={r['keyword_recall']:.4f}  | {r['question'][:40]}")

if __name__ == "__main__":
    asyncio.run(main())
