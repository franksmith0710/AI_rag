"""
切换 ONNX Embedding 后重置文档状态

用途：模型换了，向量空间变了，必须清空重新处理
执行后效果：
  1. Chroma 目录 tenant_* 全部删除
  2. documents.status 重置为 pending
  3. document_chunks 全部清空

使用：
  主机侧：python -m backend.scripts.reindex_prep
  Docker 容器内：docker exec rag-backend python -m scripts.reindex_prep
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sqlalchemy import text
from core.database import sync_engine
from core.config import get_settings

settings = get_settings()


def clear_chroma_dir():
    """删除所有 tenant Chroma 目录"""
    base = settings.chroma_persist_dir
    if not os.path.isdir(base):
        print(f"[跳过] Chroma 目录不存在: {base}")
        return 0

    count = 0
    for entry in os.listdir(base):
        if entry.startswith("tenant_"):
            full = os.path.join(base, entry)
            shutil.rmtree(full, ignore_errors=True)
            print(f"  ✓ 删除 {full}")
            count += 1
    return count


def reset_db():
    """重置文档状态 + 清空 chunks"""
    with sync_engine.begin() as conn:
        n_chunks = conn.execute(
            text("SELECT count(*) FROM document_chunks")
        ).scalar() or 0
        n_docs = conn.execute(
            text("SELECT count(*) FROM documents")
        ).scalar() or 0

        conn.execute(text("DELETE FROM document_chunks"))
        n_updated = conn.execute(
            text(
                "UPDATE documents "
                "SET status = 'pending', chunk_count = 0 "
                "WHERE status = 'completed'"
            )
        ).rowcount

    return n_docs, n_chunks, n_updated


def main():
    print("=" * 50)
    print("Reindex 准备：清空旧向量空间")
    print("=" * 50)

    print("\n[1/2] 删除 Chroma 目录...")
    cleared = clear_chroma_dir()
    print(f"  共删除 {cleared} 个 tenant 目录")

    print("\n[2/2] 重置数据库文档状态...")
    try:
        n_docs, n_chunks, n_updated = reset_db()
        print(f"  documents 总数: {n_docs}")
        print(f"  document_chunks 总数: {n_chunks}")
        print(f"  重置 documents.status: {n_updated} 条")
    except Exception as e:
        print(f"  ✗ DB 重置失败: {e}")
        print("  请检查 .env 中数据库配置是否正确")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("完成。下一步在界面上重新处理所有文档")
    print("=" * 50)


if __name__ == "__main__":
    main()
