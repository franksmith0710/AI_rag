"""
意图识别 & Query 改写模块
负责：1) 问候语/闲聊识别 2) 代词消解+上下文补全（LLM驱动）
"""

import re
from typing import List, Dict

GREETING_KEYWORDS = [
    "你好", "您好", "hello", "hi", "嗨", "哈喽", "在吗", "在不在",
    "早上好", "下午好", "晚上好", "晚安", "求助", "帮忙"
]

GREETING_PATTERNS = [
    r"^[吗呢吧呀啊哈喽]+$",
    r"^[呃嗯啊哦噢]+$",
]

MAX_GREETING_LEN = 8


def is_greeting_query(query: str) -> bool:
    """
    判断是否为问候语/闲聊 query
    """
    q = query.strip()
    if not q:
        return True
    if len(q) > MAX_GREETING_LEN:
        return False
    for kw in GREETING_KEYWORDS:
        if q.startswith(kw):
            return True
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, q):
            return True
    return False


async def rewrite_query(query: str, conversation_history: List[Dict] = None) -> str:
    """
    轻量级 Query 改写（LLM驱动，真正使用上下文）

    职责：把用户口语化、有代词、有省略的问句
         改写为一句完整、无歧义、适合检索的标准问句

    输入：原始 query + 最近5条用户问句
    输出：改写后问句 或 原始 query（失败时降级）
    """
    if not query:
        return query

    query = query.strip()
    if not conversation_history:
        return query

    history_str = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in conversation_history[-5:]
    ])

    prompt = f"""【历史用户问题】
{history_str}

【当前问题】
{query}

任务：
1. 如果当前问题包含代词（包括但不限于：这个、那个、它、此、该、他们、她们），结合历史补全主题。
2. 如果当前问题是省略句，结合历史补全为完整问句。
3. 如果当前问题清晰、无代词、无省略，直接输出原问题。
4. 不要强行关联无关历史。
5. 只输出最终检索问句，不要解释。
"""

    try:
        from core.llm_factory import create_llm
        llm = create_llm(temperature=0.1, max_tokens=128, timeout=15)
        rewritten = (await llm.ainvoke(prompt)).content.strip()
        return rewritten if rewritten else query
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"query_rewrite: 改写失败 - {e}")
        return query


async def expand_query_variants(
    query: str,
    conversation_history: List[Dict] = None,
    num_variants: int = 3,
) -> List[str]:
    """
    生成多个语义等价的检索变体，用于多路召回提升 recall。

    输入：改写后的 query + 历史
    输出：List[str]，第一个元素为原始 query，后续为 LLM 生成变体
    失败降级：返回 [query]
    """
    if not query:
        return [""]

    query = query.strip()
    if not query:
        return [query]

    history_str = ""
    if conversation_history:
        history_str = "\n".join([
            f"user: {m['content']}"
            for m in conversation_history[-5:]
        ])

    extra = num_variants - 1  # 需要额外生成几个变体
    if extra < 1:
        return [query]

    prompt = f"""【历史用户问题】
{history_str}

【当前问题】
{query}

任务：
1. 根据当前问题和历史对话，生成{extra}个语义等价但表达方式不同的检索问句。
2. 每个变体应侧重不同关键词和表述角度，以提升检索覆盖率。
3. 不要强行关联无关历史。
4. 每行一个问句，不要编号，不要多余内容。"""

    try:
        from core.llm_factory import create_llm
        llm = create_llm(temperature=0.1, max_tokens=256, timeout=15)
        text = (await llm.ainvoke(prompt)).content.strip()
        variants = [line.strip() for line in text.split("\n") if line.strip()]
        # 去重、去空
        seen = set()
        deduped = []
        for v in variants:
            if v and v not in seen:
                deduped.append(v)
                seen.add(v)
        # 用足 extra 个（不够就取全部），前面放原始 query
        result = [query] + deduped[:extra]
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"expand_query_variants: 扩展失败 - {e}")
        return [query]