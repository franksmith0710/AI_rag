"""
文本分块工具
使用 LangChain 的 RecursiveCharacterTextSplitter 进行文档分块
支持中文分词，保留句子完整性
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


def create_text_splitter(
    chunk_size: int = 650,
    chunk_overlap: int = 100,
    separators: List[str] = None
) -> RecursiveCharacterTextSplitter:
    """
    创建文本分块器

    Args:
        chunk_size: 分块大小(字符数)，默认 650
        chunk_overlap: 分块重叠字数，默认 100
        separators: 分隔符列表，按优先级排序

    Returns:
        RecursiveCharacterTextSplitter 实例
    """
    if separators is None:
        # 默认分隔符：优先段落边界，其次句子
        # \n\n 最优先，保证段落完整性；句子分隔符放后面
        separators = ["\n\n", "\n", "；", "。", "？", "！", "，", " "]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,  # 按字符数计算长度
        is_separator_regex=False  # 分隔符为普通字符串
    )


def split_text(text: str, chunk_size: int = 650, chunk_overlap: int = 100) -> List[str]:
    """
    将文本分割成 chunks

    Args:
        text: 待分割的文本
        chunk_size: 分块大小
        chunk_overlap: 分块重叠

    Returns:
        分块后的文本列表

    示例:
        >>> text = "第一段内容。\\n\\n第二段内容。"
        >>> chunks = split_text(text)
        >>> print(chunks)
        ["第一段内容。", "第二段内容。"]
    """
    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def _jieba_cut_for_bm25(text: str) -> list:
    """BM25 分词：jieba 精确模式"""
    import jieba
    return list(jieba.cut(text, cut_all=False))