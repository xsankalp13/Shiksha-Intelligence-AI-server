"""
Optional Long-term Student Memory Service — backed by Supabase PostgreSQL.

Stores persistent student preferences and learning patterns across sessions.
Completely optional — returns empty defaults if DATABASE_URL is not set
or ENABLE_LONG_TERM_MEMORY is False.

Schema (auto-created on first use):
    CREATE TABLE ai_student_profiles (
        user_id       INTEGER PRIMARY KEY,
        language_pref VARCHAR(20) DEFAULT 'English',
        weak_subjects TEXT[]      DEFAULT '{}',
        strong_subjects TEXT[]    DEFAULT '{}',
        preferred_response_length VARCHAR(10) DEFAULT 'short',
        notes         TEXT        DEFAULT '',
        updated_at    TIMESTAMP   DEFAULT NOW()
    );
"""
from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.core.logging import logger

# Default profile returned when memory is disabled or user has no record yet
_DEFAULT_PROFILE: dict[str, Any] = {
    "language_pref": "English",
    "weak_subjects": [],
    "strong_subjects": [],
    "preferred_response_length": "short",
    "notes": "",
}

_pool = None


async def _get_pool():
    """Lazy-initialised asyncpg connection pool."""
    global _pool
    if _pool is None:
        import asyncpg
        _pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=2, max_size=10)
        await _ensure_table(_pool)
    return _pool


async def _ensure_table(pool) -> None:
    """Create the profile table if it doesn't already exist."""
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_student_profiles (
                user_id                   INTEGER      PRIMARY KEY,
                language_pref             VARCHAR(20)  NOT NULL DEFAULT 'English',
                weak_subjects             TEXT[]       NOT NULL DEFAULT '{}',
                strong_subjects           TEXT[]       NOT NULL DEFAULT '{}',
                preferred_response_length VARCHAR(10)  NOT NULL DEFAULT 'short',
                notes                     TEXT         NOT NULL DEFAULT '',
                updated_at                TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """)


class ProfileService:
    """
    Async interface for reading and writing long-term student profiles.

    All methods are safe to call even when long-term memory is disabled —
    they return defaults and skip DB operations silently.
    """

    @staticmethod
    async def get_profile(user_id: int) -> dict[str, Any]:
        """
        Load a student's persistent profile.
        Returns default profile if memory is disabled or no record exists.
        """
        if not settings.ENABLE_LONG_TERM_MEMORY or not settings.DATABASE_URL:
            return dict(_DEFAULT_PROFILE)

        try:
            pool = await _get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM ai_student_profiles WHERE user_id = $1",
                    user_id,
                )
                if row:
                    return {
                        "language_pref": row["language_pref"],
                        "weak_subjects": list(row["weak_subjects"]),
                        "strong_subjects": list(row["strong_subjects"]),
                        "preferred_response_length": row["preferred_response_length"],
                        "notes": row["notes"],
                    }
        except Exception as exc:
            logger.warning("ProfileService.get_profile failed", user_id=user_id, error=str(exc))

        return dict(_DEFAULT_PROFILE)

    @staticmethod
    async def update_profile(user_id: int, updates: dict[str, Any]) -> None:
        """
        Upsert preference fields for a student.
        Only the keys present in `updates` are changed.

        Example:
            await ProfileService.update_profile(42, {"language_pref": "Hindi"})
        """
        if not settings.ENABLE_LONG_TERM_MEMORY or not settings.DATABASE_URL:
            return

        allowed_fields = {
            "language_pref", "weak_subjects", "strong_subjects",
            "preferred_response_length", "notes",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed_fields}
        if not filtered:
            return

        try:
            pool = await _get_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ai_student_profiles (user_id, {cols})
                    VALUES ($1, {placeholders})
                    ON CONFLICT (user_id) DO UPDATE
                    SET {updates}, updated_at = NOW()
                """.format(
                    cols=", ".join(filtered.keys()),
                    placeholders=", ".join(f"${i+2}" for i in range(len(filtered))),
                    updates=", ".join(
                        f"{col} = EXCLUDED.{col}" for col in filtered.keys()
                    ),
                ), user_id, *filtered.values())

        except Exception as exc:
            logger.warning("ProfileService.update_profile failed", user_id=user_id, error=str(exc))

    @staticmethod
    async def close() -> None:
        """Close the connection pool on server shutdown."""
        global _pool
        if _pool:
            await _pool.close()
            _pool = None
            logger.info("ProfileService: DB pool closed")
