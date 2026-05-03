"""
文本分块工具
使用 LangChain 的 RecursiveCharacterTextSplitter 进行文档分块
支持中文分词，保留句子完整性
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


def create_text_splitter(
    chunk_size: int = 700,
    chunk_overlap: int = 150,
    separators: List[str] = None
) -> RecursiveCharacterTextSplitter:
    """
    创建文本分块器

    Args:
        chunk_size: 分块大小(字符数)，默认 700
        chunk_overlap: 分块重叠字数，默认 150
        separators: 分隔符列表，按优先级排序

    Returns:
        RecursiveCharacterTextSplitter 实例
    """
    if separators is None:
        # 默认分隔符：按优先级从低到高
        # 先按段落分割(\n\n)，再按句子(\n，。？！)
        # 这样可以保证在段落边界分割，保留语义完整性
        separators = ["\n\n", "\n", "。", "？", "！", "；", "，", " "]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,  # 按字符数计算长度
        is_separator_regex=False  # 分隔符为普通字符串
    )


def split_text(text: str, chunk_size: int = 700, chunk_overlap: int = 150) -> List[str]:
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


def split_documents(documents: List[str], chunk_size: int = 700, chunk_overlap: int = 150) -> List[str]:
    """
    批量分割多个文档

    Args:
        documents: 文档列表
        chunk_size: 分块大小
        chunk_overlap: 分块重叠

    Returns:
        所有文档的分块结果
    """
    splitter = create_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)