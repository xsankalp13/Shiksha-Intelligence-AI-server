"""
ERP Tool — Teacher Analytics

Provides class-level (section-level) data for teachers.
These endpoints are only called when state["role"] == "TEACHER".

Endpoints assume the same JWT-authenticated ERP backend. The teacher's
JWT token carries class/section context — the backend filters accordingly.
"""
from __future__ import annotations

from app.tools.erp.base_client import ERPClient


async def fetch_class_attendance_summary(
    jwt_token: str,
    class_id: int | None = None,
    section_id: int | None = None,
) -> dict:
    """
    GET /api/v1/auth/ams/class-summary
    Returns section-level attendance aggregates:
      - Total students
      - % present today
      - Students at risk (below threshold)
      - Trend over last 30 days

    Source: ClassAttendanceController.java (confirm endpoint path)
    """
    params: dict = {}
    if class_id:
        params["classId"] = class_id
    if section_id:
        params["sectionId"] = section_id

    async with ERPClient(jwt_token) as client:
        return await client.get("/auth/ams/class-summary", params=params)


async def fetch_class_performance(
    jwt_token: str,
    class_id: int | None = None,
    section_id: int | None = None,
    exam_uuid: str | None = None,
) -> dict:
    """
    GET /api/v1/auth/examination/class-performance
    Returns section-level marks summary:
      - Subject-wise average marks
      - Top & bottom 5 students
      - Pass/fail distribution

    Source: ClassPerformanceController.java (confirm endpoint path)
    """
    params: dict = {}
    if class_id:
        params["classId"] = class_id
    if section_id:
        params["sectionId"] = section_id
    if exam_uuid:
        params["examUuid"] = exam_uuid

    async with ERPClient(jwt_token) as client:
        return await client.get("/auth/examination/class-performance", params=params)


async def fetch_at_risk_students(jwt_token: str, threshold: int = 75) -> list:
    """
    GET /api/v1/auth/ams/at-risk
    Returns students whose attendance has fallen below `threshold` percent.
    Used by the AI for proactive teacher alerts.

    Source: AttendanceRiskController.java (confirm endpoint path)
    """
    async with ERPClient(jwt_token) as client:
        return await client.get(
            "/auth/ams/at-risk",
            params={"threshold": threshold},
        )
