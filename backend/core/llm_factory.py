"""
LLM 客户端工厂模块
集中管理 ChatOpenAI 实例化，避免重复配置
"""
from langchain_openai import ChatOpenAI
from .config import get_settings

settings = get_settings()


def create_llm(
    temperature: float = 0.1,
    max_tokens: int = 256,
    timeout: int = 30,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_rewrite_model or "deepseek-chat",
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
