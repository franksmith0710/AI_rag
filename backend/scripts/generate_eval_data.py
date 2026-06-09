"""
测试数据生成脚本 v3
使用 DeepSeek V4 Flash 从 DocumentChunk 自动生成 QA 对
"""

import asyncio
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger("generate_eval_data")

# 本地运行覆盖 Docker 配置，必须在 settings 加载前设置
os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "5433"
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PASSWORD", "redis123456")
os.environ.setdefault("EMBEDDING_ONNX_PATH", r"D:\hf_models\BAAI\bge-m3-onnx\bge-m3.onnx")
os.environ.setdefault("EMBEDDING_MODEL_PATH", r"D:\hf_models\BAAI\bge-m3")
os.environ.setdefault("RERANKER_ONNX_PATH", r"D:\hf_models\BAAI\bge-reranker-v2-m3-onnx\bge-reranker-v2-m3.onnx")
os.environ.setdefault("RERANKER_MODEL_PATH", r"D:\hf_models\BAAI\bge-reranker-v2-m3")
os.environ.setdefault("CHROMA_PERSIST_DIR", r"D:\School\AI_rag\backend\vector_store\chroma")
os.environ.setdefault("UPLOAD_DIR", r"D:\School\AI_rag\backend\uploads")
os.environ.setdefault("JWT_SECRET_KEY", "your-secret-key-change-in-production")

from core.config import get_settings
from openai import AsyncOpenAI

settings = get_settings()

# 清除可能干扰 openai 客户端的 env var
for _env in ("LANGCHAIN_TRACING_V2", "OPENAI_API_KEY"):
    os.environ.pop(_env, None)

from core.database import async_session_maker, init_db
from sqlalchemy import select
from models.db_models import Document, DocumentChunk

TEST_DATA_PATH = Path(__file__).parent / "test_data.json"
MAX_CHUNKS = 50
CONCURRENCY = 3

SYSTEM_PROMPT = """你是一个数据标注专家。请根据用户提供的文本，生成3个自然的用户提问，并给出基于文本的标准答案。

要求：
- 问题要自然，像真实用户会问的
- 答案要基于文本内容，简洁准确
- 3个问题要覆盖不同的角度（概念、功能、细节等）
- 输出必须是 JSON 数组格式，不要包含其他内容

输出格式：
[
  {"question": "问题1", "answer": "答案1"},
  {"question": "问题2", "answer": "答案2"},
  {"question": "问题3", "answer": "答案3"}
]"""


async def generate_qa(text: str, sem: asyncio.Semaphore) -> list[dict] | None:
    async with sem:
        client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        try:
            result = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"文本：\n{text}"},
                ],
                temperature=0.3,
                max_tokens=1024,
                timeout=30,
            )
            raw = result.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removesuffix("```").strip()
            pairs = json.loads(raw)
            if not isinstance(pairs, list):
                raise ValueError("not a list")
            for p in pairs:
                if not isinstance(p.get("question"), str) or not isinstance(p.get("answer"), str):
                    raise ValueError("invalid item")
            return pairs
        except Exception as e:
            text_preview = text[:40].replace("\n", " ")
            logger.warning(f"  LLM 生成失败: {e} | text={text_preview}")
            return None


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

    logger.info(f"数据库共 {len(rows)} 个 chunks，处理前 {MAX_CHUNKS} 个")

    chunks_to_process = []
    for chunk, doc_title in rows:
        text = chunk.text.strip()
        if not text or len(text) < 20:
            continue
        chunks_to_process.append((chunk, doc_title))
        if len(chunks_to_process) >= MAX_CHUNKS:
            break

    logger.info(f"有效 chunks: {len(chunks_to_process)}")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [generate_qa(chunk.text.strip(), sem) for chunk, _ in chunks_to_process]

    results_list = await asyncio.gather(*tasks)

    test_data = []
    success = 0
    for (chunk, doc_title), pairs in zip(chunks_to_process, results_list):
        if pairs is None:
            continue
        for pair in pairs:
            test_data.append({
                "chunk_id": chunk.id,
                "document_title": doc_title,
                "question": pair["question"],
                "ground_truth": pair["answer"],
            })
            success += 1

    if not test_data:
        logger.error("所有 LLM 生成均失败")
        sys.exit(1)

    test_data.sort(key=lambda x: x["chunk_id"])

    TEST_DATA_PATH.write_text(
        json.dumps(test_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"生成完成: {len(test_data)} 条 QA, 成功处理 {success} 个 chunks")
    print(f"  已保存: {TEST_DATA_PATH}")

    for item in test_data[:5]:
        q = item['question'][:60]
        try:
            print(f"  Q: {q}")
        except UnicodeEncodeError:
            print(f"  Q: (unicode text)")

    logger.info(f"  共消耗 tokens: {success * 3} 次 API 调用")


if __name__ == "__main__":
    asyncio.run(generate())
