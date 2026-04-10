"""ERP Tool — Student Marks (/auth/examination/schedules/{id}/marks)"""
from app.tools.erp.base_client import ERPClient


async def fetch_marks_by_schedule(jwt_token: str, schedule_id: int) -> list:
    """
    Calls GET /api/v1/auth/examination/schedules/{scheduleId}/marks
    Source: StudentMarkController.java
    """
    async with ERPClient(jwt_token) as client:
        return await client.get(f"/auth/examination/schedules/{schedule_id}/marks")


async def fetch_exam_schedules(jwt_token: str, exam_uuid: str) -> list:
    """
    Calls GET /api/v1/auth/examination/exams/{examUuid}/schedules
    Source: ExamScheduleController.java
    """
    async with ERPClient(jwt_token) as client:
        return await client.get(f"/auth/examination/exams/{exam_uuid}/schedules")
