"""
Redis 连接模块
负责 Redis 连接管理，支持生产环境使用 Redis
开发环境自动降级为内存缓存
"""
import random
import time
import redis.asyncio as redis
from .config import get_settings
from .logging_config import setup_logging
from typing import Optional
import json

logger = setup_logging("redis_conn")
settings = get_settings()

redis_client: Optional[redis.Redis] = None
_memory_cache = {}


async def get_redis() -> Optional[redis.Redis]:
    """
    获取 Redis 客户端
    开发模式下(无 Redis 服务)返回 None，自动使用内存缓存

    Returns:
        Redis 客户端实例，或 None (开发模式)
    """
    global redis_client

    # 开发模式(Chroma)不使用 Redis，生产模式使用 Redis
    if settings.vector_store == "chroma":
        return None

    try:
        if redis_client is None:
            redis_kwargs = {
                "host": settings.redis_host,
                "port": settings.redis_port,
                "decode_responses": True,
                "encoding": "utf-8"
            }
            if settings.redis_password:
                redis_kwargs["password"] = settings.redis_password
            redis_client = redis.Redis(**redis_kwargs)
            await redis_client.ping()
        return redis_client
    except Exception as e:
        logger.warning(f"Redis 连接失败: {e}，使用内存缓存")
        return None


async def get_cached(key: str) -> Optional[str]:
    """
    获取缓存值 (统一接口)

    Args:
        key: 缓存键

    Returns:
        缓存值字符串，或 None
    """
    r = await get_redis()
    if r:
        # 使用 Redis
        return await r.get(key)

    # 使用内存缓存兜底（带 TTL）
    entry = _memory_cache.get(key)
    if entry is None:
        return None
    if time.time() > entry["expire"]:
        del _memory_cache[key]
        return None
    return entry["value"]


async def set_cached(key: str, value: str, expire: int = 86400):
    """
    设置缓存值 (统一接口)

    Args:
        key: 缓存键
        value: 缓存值
        expire: 过期时间(秒)，默认 24h，实际增加随机抖动防雪崩
    """
    r = await get_redis()
    if r:
        # 增加 ±30% 随机抖动，防止大量 key 同时过期
        actual_expire = expire + random.randint(0, expire // 3)
        await r.setex(key, actual_expire, value)
    else:
        # 使用内存缓存兜底（带 TTL）
        _memory_cache[key] = {"value": value, "expire": time.time() + expire}


async def delete_cached(key: str):
    """
    删除缓存 (统一接口)

    Args:
        key: 缓存键
    """
    r = await get_redis()
    if r:
        await r.delete(key)
    else:
        _memory_cache.pop(key, None)


async def close_redis():
    """
    关闭 Redis 连接
    应用关闭时调用
    """
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None