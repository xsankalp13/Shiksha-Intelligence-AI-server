"""
Base async HTTP client for ERP communication.
- Injects the student's JWT token automatically
- Handles retries using tenacity
- All methods return parsed dict/list or raise HTTPException
"""
from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.config import settings
from app.core.logging import logger


class ERPClient:
    """
    Async HTTP client scoped to a single request/user.
    Pass the raw JWT string (without 'Bearer ' prefix).
    """

    def __init__(self, jwt_token: str):
        self._headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        }
        self._base = settings.erp_api_root
        self._client = httpx.AsyncClient(
            base_url=self._base,
            headers=self._headers,
            timeout=httpx.Timeout(10.0, connect=5.0),
        )

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET with retry on transient network errors."""
        url = path
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type(httpx.TransportError),
            ):
                with attempt:
                    resp = await self._client.get(url, params=params)
                    resp.raise_for_status()
                    logger.debug("ERP GET", path=path, status=resp.status_code)
                    return resp.json()

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "ERP HTTP error", path=path,
                status=exc.response.status_code,
                body=exc.response.text[:200],
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"ERP error on {path}: {exc.response.text[:200]}",
            )
        except Exception as exc:
            logger.error("ERP unexpected error", path=path, error=str(exc))
            raise HTTPException(status_code=502, detail=f"ERP unreachable: {exc}")

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ERPClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
