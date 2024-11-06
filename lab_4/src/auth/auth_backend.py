import redis.asyncio

from fastapi_users.authentication import (BearerTransport, 
                                          RedisStrategy, 
                                          AuthenticationBackend)

from config import REDIS_HOST, REDIS_PORT


bearer_transport = BearerTransport(tokenUrl="auth/login")

redis = redis.asyncio.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", 
                               decode_responses=True)

def get_redis_strategy() -> RedisStrategy:
    return RedisStrategy(redis, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(
    name="redis_auth_backend",
    transport=bearer_transport,
    get_strategy=get_redis_strategy,
)
