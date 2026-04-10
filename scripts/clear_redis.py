import asyncio
import redis.asyncio as aioredis
from app.core.config import settings

async def clear_redis_config():
    pool = aioredis.ConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)
    client = aioredis.Redis(connection_pool=pool)
    await client.delete(settings.REDIS_AI_CONFIG_KEY)
    print("Deleted key:", settings.REDIS_AI_CONFIG_KEY)
    await client.aclose()

if __name__ == "__main__":
    asyncio.run(clear_redis_config())
