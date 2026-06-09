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


async def rewrite_and_expand(
    query: str,
    conversation_history: List[Dict] = None,
    num_variants: int = 3,
) -> tuple:
    """
    合并版：先指代消解改写，再基于改写结果生成语义变体。
    一次 LLM 调用完成，省一次往返。
    返回: (rewritten_query, [rewritten_query, variant1, variant2, ...])
    """
    if not query:
        return "", [""]

    query = query.strip()
    if not conversation_history:
        # 没有历史，不需要改写，直接基于 query 做变体
        variants = await expand_query_variants(query, None, num_variants)
        return query, variants

    history_str = "\n".join([
        f"user: {m['content']}"
        for m in conversation_history[-5:]
    ])

    extra = num_variants - 1
    if extra < 1:
        # 不需要变体，只用 rewrite
        rewritten = await rewrite_query(query, conversation_history)
        return rewritten, [rewritten]

    prompt = f"""【对话历史】
{history_str}

【当前用户问题】
{query}

第一步：指代消解
- 如果当前问题含有代词（这个、那个、它、此、该、他们、她们等），结合对话历史补全指代对象
- 如果当前问题是省略句（缺少主语或关键信息），结合对话历史补全为完整问句
- 如果当前问题本身清晰完整、无代词、无省略，则直接使用原问题作为【标准问句】

第二步：变体生成
基于第一步得到的【标准问句】，将其改写为{extra}个语义等价但表达方式不同的检索变体。
变体必须基于【标准问句】，而不是原始问题。
- 每个变体替换关键词同义词或调整句式结构
- 不要改变问题的核心语义和意图
- 不要引入对话历史中的新话题
- 不要复述标准问句本身

输出格式（严格每行一句，不要编号、不要引号、不要多余文字）：
标准问句
变体1
变体2
..."""

    try:
        from core.llm_factory import create_llm
        llm = create_llm(temperature=0.1, max_tokens=256, timeout=15)
        text = (await llm.ainvoke(prompt)).content.strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        rewritten = lines[0] if lines else query
        seen = {rewritten}
        variants = [rewritten]
        for v in lines[1:]:
            if v and v not in seen:
                variants.append(v)
                seen.add(v)
        result = variants[:num_variants]
        # 如果不够 num_variants，用第一个变体补充
        while len(result) < num_variants:
            result.append(result[-1])
        return rewritten, result
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"rewrite_and_expand: 失败 - {e}")
        # 降级：分开调用
        rewritten = await rewrite_query(query, conversation_history)
        variants = await expand_query_variants(rewritten, conversation_history, num_variants)
        return rewritten, variants


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