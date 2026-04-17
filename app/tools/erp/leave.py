"""
ERP Tool — Leave Management

Supports both reading existing leave records and submitting new applications.

NOTE: The POST endpoint is set to /auth/leave/apply as a placeholder.
      Confirm with your Java LeaveController before enabling SUBMIT_LEAVE intent.
"""
from __future__ import annotations

from typing import Any

from app.tools.erp.base_client import ERPClient


async def fetch_leave_records(
    jwt_token: str,
    page: int = 0,
    size: int = 10,
) -> dict:
    """
    GET /api/v1/auth/leave/records
    Returns the student's existing leave applications and their approval status.
    """
    async with ERPClient(jwt_token) as client:
        return await client.get(
            "/auth/leave/records",
            params={"page": page, "size": size, "sort": "createdAt,desc"},
        )


async def submit_leave_application(
    jwt_token: str,
    payload: dict[str, Any],
) -> dict:
    """
    POST /api/v1/auth/leave/apply
    Submits a leave application on behalf of the student.

    Expected payload:
        {
            "fromDate":  "2024-11-20",
            "toDate":    "2024-11-21",
            "reason":    "Medical — fever",
            "leaveType": "MEDICAL"   # MEDICAL | PERSONAL | FAMILY
        }

    PLACEHOLDER: Confirm endpoint path with Java LeaveController before use.
    """
    async with ERPClient(jwt_token) as client:
        # ERPClient only has GET; add POST support inline
        resp = await client._client.post("/auth/leave/apply", json=payload)
        resp.raise_for_status()
        return resp.json()
