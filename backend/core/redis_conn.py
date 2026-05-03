"""
Redis 连接模块
负责 Redis 连接管理，支持生产环境使用 Redis
开发环境自动降级为内存缓存
"""
import redis.asyncio as redis
from .config import get_settings
from typing import Optional
import json

settings = get_settings()

redis_client: Optional[redis.Redis] = None  # Redis 客户端单例
_memory_cache = {}  # 内存缓存兜底方案


async def get_redis() -> Optional[redis.Redis]:
    """
    获取 Redis 客户端
    开发模式下(无 Redis 服务)返回 None，自动使用内存缓存

    Returns:
        Redis 客户端实例，或 None (开发模式)
    """
    global redis_client

    # 开发模式(sqLite)不使用 Redis
    if settings.db_mode == "sqlite":
        return None

    try:
        if redis_client is None:
            redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True,  # 返回字符串而非字节
                encoding="utf-8"
            )
            # 测试连接
            await redis_client.ping()
        return redis_client
    except Exception as e:
        print(f"Redis 连接失败: {e}，使用内存缓存")
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

    # 使用内存缓存兜底
    if key in _memory_cache:
        return _memory_cache[key]
    return None


async def set_cached(key: str, value: str, expire: int = 3600):
    """
    设置缓存值 (统一接口)

    Args:
        key: 缓存键
        value: 缓存值
        expire: 过期时间(秒)，默认 1 小时
    """
    r = await get_redis()
    if r:
        # 使用 Redis
        await r.setex(key, expire, value)
    else:
        # 使用内存缓存兜底
        _memory_cache[key] = value


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