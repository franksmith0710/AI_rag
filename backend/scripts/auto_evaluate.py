"""
一键自动评估脚本
流程：生成测试数据 → 运行 RAGAS 评估 → 输出报告
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main():
    print("=" * 60)
    print("  RAGAS 一键自动评估")
    print("=" * 60)

    # Step 1: 生成测试数据
    print("\n[1/2] 生成测试数据...")
    from scripts.generate_eval_data import generate
    await generate()

    # Step 2: 运行评估
    print("\n[2/2] 运行 RAGAS 评估...")
    from scripts.evaluate import main as run_eval
    await run_eval()

    print("\n" + "=" * 60)
    print("  评估完成！")
    print(f"  报告路径: {Path(__file__).parent / 'eval_report.json'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
