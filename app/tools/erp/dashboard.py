"""ERP Tool — Student Dashboard Intelligence (/student/dashboard/intelligence)"""
from app.tools.erp.base_client import ERPClient


async def fetch_dashboard_intelligence(jwt_token: str) -> dict:
    """
    Calls GET /api/v1/student/dashboard/intelligence
    Returns: IntelligenceResponseDTO (profile, academicPulse, financeHealth, activityFeed)
    Source: StudentDashboardController.java
    """
    async with ERPClient(jwt_token) as client:
        return await client.get("/student/dashboard/intelligence")
