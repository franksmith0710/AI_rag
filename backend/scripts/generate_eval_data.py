"""
测试数据生成脚本
从数据库中的 DocumentChunk 自动生成 QA 对，用于 RAGAS 评估
"""

import asyncio
import json
import sys
import os
from pathlib import Path

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
logger = logging.getLogger("generate_eval_data")

from core.database import async_session_maker, init_db
from sqlalchemy import select
from models.db_models import Document, DocumentChunk

TEST_DATA_PATH = Path(__file__).parent / "test_data.json"


def extract_question_from_text(text: str) -> str:
    """从文本中提取第一句话作为问题"""
    text = text.strip()
    for sep in ["。", "！", "？", ".\n", ".\r\n", ". ", "\n"]:
        idx = text.find(sep)
        if 0 < idx < 200:
            return text[:idx] + "？"
    if len(text) > 100:
        return text[:100] + "？"
    return text + "？" if text else ""


async def generate():
    await init_db()

    async with async_session_maker() as db:
        result = await db.execute(
            select(DocumentChunk, Document.title)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.status == "completed")
            .order_by(Document.id, DocumentChunk.chunk_index)
        )
        rows = result.all()

    if not rows:
        logger.error("数据库中没有已处理的文档 chunks，请先上传并处理文档")
        sys.exit(1)

    test_data = []
    for chunk, doc_title in rows:
        text = chunk.text.strip()
        if not text or len(text) < 20:
            continue

        question = extract_question_from_text(text)
        test_data.append({
            "chunk_id": chunk.id,
            "document_title": doc_title,
            "question": question,
            "ground_truth": text,
        })

    test_data.sort(key=lambda x: x["chunk_id"])

    TEST_DATA_PATH.write_text(
        json.dumps(test_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"生成完成: {len(test_data)} 条")
    print(f"  已保存: {TEST_DATA_PATH}")

    for item in test_data[:5]:
        q = item['question'][:50]
        try:
            print(f"  [{item['chunk_id']}] {q}")
        except UnicodeEncodeError:
            print(f"  [{item['chunk_id']}] (unicode text)")
    if len(test_data) > 5:
        print(f"  ... total {len(test_data)} items")


if __name__ == "__main__":
    asyncio.run(generate())
