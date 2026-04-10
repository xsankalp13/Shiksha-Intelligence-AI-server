"""
Redis-backed session service.

Responsibilities:
1. Storing conversation history (last N turns verbatim + compressed older summary)
2. Storing/retrieving the active AI config
3. Providing a simple ping for health checks
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import logger

# ── Module-level connection pool ──────────────────────────────────────────────
_pool: aioredis.ConnectionPool | None = None
_client: aioredis.Redis | None = None


def _get_client() -> aioredis.Redis:
    global _pool, _client
    if _client is None:
        _pool = aioredis.ConnectionPool.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=20,
        )
        _client = aioredis.Redis(connection_pool=_pool)
    return _client


# ── Constants ─────────────────────────────────────────────────────────────────
SESSION_PREFIX = "ai:session:"
MAX_VERBATIM_TURNS = 6           # Keep last 6 turns (3 user + 3 assistant) verbatim
SUMMARY_KEY_SUFFIX = ":summary"  # Older context compressed here


class SessionService:

    # ── Connection Management ─────────────────────────────────────────────

    @staticmethod
    async def ping() -> None:
        client = _get_client()
        await client.ping()
        logger.info("Redis connection established", url=settings.REDIS_URL)

    @staticmethod
    async def close() -> None:
        global _client, _pool
        if _client:
            await _client.aclose()
            _client = None
        logger.info("Redis connection closed")

    # ── Session (Conversation History) ────────────────────────────────────

    @staticmethod
    def new_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    async def get_history(session_id: str) -> dict[str, Any]:
        """
        Returns:
          {
            "turns": [ {"role": "user"|"assistant", "content": "..."}, ... ],  # last N verbatim
            "summary": "..."  # compressed older context (or "")
          }
        """
        client = _get_client()
        key = f"{SESSION_PREFIX}{session_id}"
        raw = await client.get(key)
        if raw:
            return json.loads(raw)
        return {"turns": [], "summary": ""}

    @staticmethod
    async def append_turn(session_id: str, role: str, content: str) -> None:
        """
        Appends a turn and trims older messages.
        If the history exceeds MAX_VERBATIM_TURNS, the oldest pair is
        dropped (a future enhancement can compress it into the summary).
        """
        client = _get_client()
        key = f"{SESSION_PREFIX}{session_id}"

        data = await SessionService.get_history(session_id)
        data["turns"].append({"role": role, "content": content})

        # Trim — always keep turns in pairs (user + assistant)
        while len(data["turns"]) > MAX_VERBATIM_TURNS:
            data["turns"].pop(0)  # Drop oldest

        await client.setex(
            key,
            settings.REDIS_SESSION_TTL_SECONDS,
            json.dumps(data),
        )

    # ── AI Config ─────────────────────────────────────────────────────────

    @staticmethod
    async def get_ai_config() -> dict[str, Any] | None:
        client = _get_client()
        raw = await client.get(settings.REDIS_AI_CONFIG_KEY)
        if raw:
            return json.loads(raw)
        return None

    @staticmethod
    async def set_ai_config(config: dict[str, Any]) -> None:
        client = _get_client()
        await client.set(
            settings.REDIS_AI_CONFIG_KEY,
            json.dumps(config),
            # Config has no expiry — persists until manually changed
        )
