"""ERP Tool — Student Attendance Records (/auth/ams/records)"""
from app.tools.erp.base_client import ERPClient


async def fetch_attendance(
    jwt_token: str,
    student_id: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    page: int = 0,
    size: int = 30,
) -> dict:
    """
    Calls GET /api/v1/auth/ams/records with optional filters.
    Source: StudentAttendanceController.java
    """
    params: dict = {"page": page, "size": size, "sort": "id,desc"}
    if student_id:
        params["studentId"] = student_id
    if from_date:
        params["fromDate"] = from_date
    if to_date:
        params["toDate"] = to_date

    async with ERPClient(jwt_token) as client:
        return await client.get("/auth/ams/records", params=params)
