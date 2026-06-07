"""
从 DB 读取所有 chunks，调 DeepSeek 为每条 chunk 生成一个提问，
输出 test_data.json（可直接被 evaluate.py 消费）
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
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
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("generate_test_data")

from core.config import get_settings
from core.database import async_session_maker
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from models.db_models import DocumentChunk, Document
from langchain_openai import ChatOpenAI

settings = get_settings()

OUTPUT_PATH = Path(__file__).parent / "test_data.json"

GENERATION_PROMPT = """请你根据以下文本片段，提出一个具体的、有准确答案的问题。

要求：
1. 问题必须能在该文本片段中找到明确答案，不能超出原文范围
2. 问题要具体，不要泛泛而问（不要问"本文讲了什么"这类笼统问题）
3. 只输出问题本身，不要输出答案、不要加引号、不要加任何额外文字

文本片段：
{chunk_text}

问题："""


def build_llm(temperature: float = 0.1):
    return ChatOpenAI(
        model=settings.deepseek_model or "deepseek-chat",
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        max_tokens=64,
        timeout=30,
    )


async def generate_question(chunk_text: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        llm = build_llm()
        prompt = GENERATION_PROMPT.format(chunk_text=chunk_text)
        try:
            response = await llm.ainvoke(prompt)
            question = response.content.strip().strip('"').strip("'").strip()
            return question
        except Exception as e:
            logger.warning(f"生成问题失败: {e}")
            return ""


async def main():
    logger.info("正在连接数据库并读取 chunks ...")
    em = None
    try:
        from core.chroma_conn import get_embedding_model
        em = get_embedding_model()
    except Exception:
        logger.warning("嵌入模型预热失败，继续运行（仅用于文本生成无需嵌入）")

    chunks = []
    async with async_session_maker() as db:
        result = await db.execute(
            select(DocumentChunk)
            .options(joinedload(DocumentChunk.document))
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )
        chunks = result.scalars().all()

    if not chunks:
        logger.warning("数据库中没有 chunk，请先上传并处理文档")
        print("[]", file=open(OUTPUT_PATH, "w", encoding="utf-8"))
        return

    logger.info(f"共 {len(chunks)} 条 chunk，开始生成问题（并发 5）...")

    sem = asyncio.Semaphore(5)
    test_data = []
    tasks = []

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.text.strip() if chunk.text else ""
        if len(chunk_text) < 20:
            logger.info(f"  [{i+1}/{len(chunks)}] 跳过过短 chunk id={chunk.id}")
            continue

        tasks.append((i, chunk, generate_question(chunk_text, sem)))

    completed = 0
    for i, chunk, coro in tasks:
        question = await coro
        completed += 1
        if not question:
            logger.info(f"  [{completed}/{len(tasks)}] 生成失败，跳过 chunk id={chunk.id}")
            continue

        doc_title = chunk.document.title if chunk.document else ""

        test_data.append({
            "chunk_id": chunk.id,
            "document_title": doc_title,
            "question": question,
            "ground_truth": chunk.text,
        })
        logger.info(f"  [{completed}/{len(tasks)}] Q: {question[:50]}...")

    test_data.sort(key=lambda x: x["chunk_id"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"  生成完成: {len(test_data)} 条")
    print(f"  输出文件: {OUTPUT_PATH}")
    print("=" * 60)
    for item in test_data:
        print(f"  [{item['document_title']}] {item['question'][:60]}")


if __name__ == "__main__":
    asyncio.run(main())
