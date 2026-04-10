"""ERP Tool — Student Finance / Invoices (/auth/finance/invoices)"""
from app.tools.erp.base_client import ERPClient


async def fetch_invoices(jwt_token: str, page: int = 0, size: int = 10) -> dict:
    """
    Calls GET /api/v1/auth/finance/invoices (paginated)
    Source: InvoiceController.java
    """
    async with ERPClient(jwt_token) as client:
        return await client.get(
            "/auth/finance/invoices",
            params={"page": page, "size": size, "sort": "issueDate,desc"},
        )
