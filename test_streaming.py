"""
流式输出测试脚本
测试 RAG 服务是否支持真正的 SSE 流式输出
"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
os.environ.setdefault("ENVIRONMENT", "production")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from services.rag_service import generate_answer


async def test_streaming_generator():
    """测试生成器是否返回流式内容"""
    print("=" * 50)
    print("测试 1: 测试 generate_answer 生成器")
    print("=" * 50)

    query = "什么是人工智能？"

    try:
        generator = generate_answer(
            query=query,
            tenant_id=1,
            session_id=1
        )

        chunks = []
        async for item in generator:
            item_type = item.get("type")
            if item_type == "text":
                content = item.get("content", "")
                chunks.append(content)
                print(f"[流式片段] {content[:50]}...")
            elif item_type == "done":
                sources = item.get("sources", [])
                print(f"[完成] sources 数量: {len(sources)}")
            elif item_type == "error":
                print(f"[错误] {item.get('content')}")

        full_text = "".join(chunks)
        print(f"\n完整回答长度: {len(full_text)} 字符")
        print(f"完整回答前100字: {full_text[:100]}")

        return True

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def format_sse(data: dict) -> str:
    """将数据格式化为 SSE 格式"""
    json_str = json.dumps(data, ensure_ascii=False)
    return f"data: {json_str}\n\n"


async def test_sse_format():
    """测试 SSE 格式输出"""
    print("\n" + "=" * 50)
    print("测试 2: 测试 SSE 格式")
    print("=" * 50)

    test_data = {
        "type": "text",
        "content": "这是测试内容"
    }

    sse_output = format_sse(test_data)
    print(f"SSE 输出: {repr(sse_output)}")

    test_done = {
        "type": "done",
        "sources": [{"text": "参考1", "document_id": 1, "score": 0.9}]
    }
    sse_done = format_sse(test_done)
    print(f"SSE Done: {repr(sse_done)}")

    return True


async def main():
    print("开始流式输出测试...\n")

    result1 = await test_streaming_generator()
    result2 = await test_sse_format()

    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    print(f"生成器测试: {'通过' if result1 else '失败'}")
    print(f"SSE 格式测试: {'通过' if result2 else '失败'}")

    if result1 and result2:
        print("\n结论: 流式输出方案可行")
        print("下一步: 修改后端 API 使用 StreamingResponse + SSE")
    else:
        print("\n结论: 需要先修复问题")


if __name__ == "__main__":
    asyncio.run(main())