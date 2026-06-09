"""
RAGAS 评估脚本 v3
纯本地评估：SemanticSimilarity + RougeScore + CHRFScore
无需 LLM API 调用，使用本地 BGE-M3 embedding 模型
"""

import asyncio
import json
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
from core.chroma_conn import get_embedding_model
from sqlalchemy import select
from models.db_models import User, Session

settings = get_settings()

TEST_DATA_JSON = Path(__file__).parent / "test_data.json"
REPORT_PATH = Path(__file__).parent / "eval_report.json"


class RagasEmbeddingWrapper:
    """将 LangChain embedding 模型适配为 RAGAS embedding 接口"""

    def __init__(self, langchain_embeddings):
        self._embeddings = langchain_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings.aembed_documents(texts)


def load_test_data() -> List[dict]:
    if not TEST_DATA_JSON.exists():
        logger.error(f"未找到测试文件: {TEST_DATA_JSON}")
        logger.error("请先运行 python scripts/generate_eval_data.py 生成测试数据")
        sys.exit(1)
    with open(TEST_DATA_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)
    logger.info(f"加载 JSON 测试数据: {len(rows)} 条")
    return rows


async def run_rag(query: str, tenant_id: int, db) -> tuple[str, list[str]]:
    from services.rag_service import chat_with_rag

    user_result = await db.execute(
        select(User).where(User.tenant_id == tenant_id).limit(1)
    )
    user = user_result.scalar_one_or_none()
    if not user:
        logger.error(f"tenant={tenant_id} 下无用户")
        return "", []

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


async def find_eval_tenant(db) -> int:
    result = await db.execute(select(User.tenant_id).limit(1))
    row = result.first()
    if row:
        return row[0]
    logger.error("数据库中无用户，请先注册")
    sys.exit(1)


async def main():
    from ragas.metrics.collections import SemanticSimilarity, RougeScore, CHRFScore

    logger.info("初始化数据库...")
    await init_db()

    logger.info("预热 GPU 模型...")
    em = get_embedding_model()
    from utils.rerank import _get_reranker
    _get_reranker()
    import jieba
    jieba.initialize()

    # 创建 RAGAS embedding wrapper
    embeddings = RagasEmbeddingWrapper(em)

    # 创建指标实例（全部本地，无需 LLM）
    semantic_metric = SemanticSimilarity(embeddings=embeddings)
    rouge_metric = RougeScore(rouge_type="rougeL", mode="fmeasure")
    chrf_metric = CHRFScore()

    test_data = load_test_data()

    async with async_session_maker() as db:
        tenant_id = await find_eval_tenant(db)
        logger.info(f"使用 tenant_id={tenant_id} 进行评估")

        dataset_samples = []
        skipped = 0

        for i, row in enumerate(test_data):
            question = row["question"]
            ground_truth = row.get("ground_truth", "")

            logger.info(f"[{i+1}/{len(test_data)}] {question[:40]}")
            answer, chunks = await run_rag(question, tenant_id, db)

            if not answer or not chunks:
                logger.warning(f"  跳过（无结果）: {question[:40]}")
                skipped += 1
                continue

            dataset_samples.append({
                "user_input": question,
                "retrieved_contexts": chunks,
                "response": answer,
                "reference": ground_truth,
            })
            logger.info(f"  answer_len={len(answer)}, chunks={len(chunks)}")

    if not dataset_samples:
        logger.error("无有效评估数据")
        sys.exit(1)

    logger.info(f"有效数据 {len(dataset_samples)} 条，跳过 {skipped} 条")

    # 逐条评分
    logger.info("运行 RAGAS 评估...")
    results_detail = []
    all_semantic = []
    all_rouge = []
    all_chrf = []

    for sample in dataset_samples:
        response = sample["response"]
        reference = sample["reference"]

        s_result = await semantic_metric.ascore(
            user_input=sample["user_input"],
            response=response,
            reference=reference,
        )
        r_result = await rouge_metric.ascore(
            user_input=sample["user_input"],
            response=response,
            reference=reference,
        )
        c_result = await chrf_metric.ascore(
            user_input=sample["user_input"],
            response=response,
            reference=reference,
        )

        s_val = float(s_result)
        r_val = float(r_result)
        c_val = float(c_result)

        all_semantic.append(s_val)
        all_rouge.append(r_val)
        all_chrf.append(c_val)

        results_detail.append({
            "question": sample["user_input"],
            "ground_truth": sample["reference"][:200],
            "answer": sample["response"],
            "chunks_count": len(sample["retrieved_contexts"]),
            "semantic_similarity": round(s_val, 4),
            "rouge_score": round(r_val, 4),
            "chrf_score": round(c_val, 4),
        })

    avg_semantic = sum(all_semantic) / len(all_semantic) if all_semantic else 0
    avg_rouge = sum(all_rouge) / len(all_rouge) if all_rouge else 0
    avg_chrf = sum(all_chrf) / len(all_chrf) if all_chrf else 0

    report = {
        "total": len(results_detail),
        "skipped": skipped,
        "avg_semantic_similarity": round(avg_semantic, 4),
        "avg_rouge_score": round(avg_rouge, 4),
        "avg_chrf_score": round(avg_chrf, 4),
        "results": results_detail,
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"  RAGAS 评估报告 ({len(results_detail)} 条, 跳过 {skipped} 条)")
    print("=" * 60)
    print(f"  Semantic Similarity:  {avg_semantic:.4f}")
    print(f"  Rouge Score:          {avg_rouge:.4f}")
    print(f"  CHRF Score:           {avg_chrf:.4f}")
    print(f"  报告已保存:           {REPORT_PATH}")
    print("=" * 60)

    for r in results_detail:
        avg = (r["semantic_similarity"] + r["rouge_score"] + r["chrf_score"]) / 3
        flag = "[OK]" if avg >= 0.7 else "[WARN]" if avg >= 0.4 else "[FAIL]"
        q = r['question'][:40]
        try:
            print(f"  {flag} S={r['semantic_similarity']:.2f} R={r['rouge_score']:.2f} C={r['chrf_score']:.2f}  | {q}")
        except UnicodeEncodeError:
            print(f"  {flag} S={r['semantic_similarity']:.2f} R={r['rouge_score']:.2f} C={r['chrf_score']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
