"""
Node 2 — Tool Executor

Calls ONLY the ERP tool(s) relevant to the classified intent.
This is the core of the "lazy fetch" token-saving strategy:
  ATTENDANCE → fetch /auth/ams/records
  FINANCE    → fetch /auth/finance/invoices
  EXAM       → fetch /auth/examination/schedules/.../marks
  SCHEDULE | GENERAL → fetch /student/dashboard/intelligence (has everything)
  RAG        → query ChromaDB (Phase 2 — skipped for now)
  LEAVE_REQUEST → no ERP call needed; LLM drafts from prompt
  UNKNOWN    → no ERP call; ask for clarification

After fetching, raw data is compressed by the Summarizer before
being placed into state["data_summary"] for the next node.
"""
from __future__ import annotations

import asyncio

from app.agents.state import AgentState
from app.core.logging import logger
from app.services import summarizer
from app.tools.erp import dashboard, attendance, finance


async def tool_executor_node(state: AgentState) -> AgentState:
    intent = state["intent"]
    jwt = state["jwt_token"]
    fetched: dict = {}
    sources: list[str] = []
    data_summary = ""

    # ── Route to the right tool ───────────────────────────────────────────
    try:
        if intent in ("SCHEDULE", "GENERAL"):
            data = await dashboard.fetch_dashboard_intelligence(jwt)
            fetched["dashboard"] = data
            sources.append("/student/dashboard/intelligence")
            data_summary = summarizer.summarize_dashboard_intelligence(data)

        elif intent == "ATTENDANCE":
            raw = await attendance.fetch_attendance(jwt, size=30)
            records = raw.get("content", []) if isinstance(raw, dict) else raw
            fetched["attendance"] = records
            sources.append("/auth/ams/records")
            data_summary = summarizer.summarize_attendance_records(records)

        elif intent == "FINANCE":
            raw = await finance.fetch_invoices(jwt)
            invoices = raw.get("content", []) if isinstance(raw, dict) else raw
            fetched["invoices"] = invoices
            sources.append("/auth/finance/invoices")
            data_summary = summarizer.summarize_invoices(invoices)

        elif intent == "EXAM":
            # Without a specific exam/schedule ID, fetch dashboard for context
            # Phase 2: derive schedule_id from conversation context
            data = await dashboard.fetch_dashboard_intelligence(jwt)
            fetched["dashboard"] = data
            sources.append("/student/dashboard/intelligence")
            data_summary = summarizer.summarize_dashboard_intelligence(data)
            data_summary += "\n\n(Ask the student for a specific exam name to get detailed marks.)"

        elif intent == "LEAVE_REQUEST":
            # No ERP call — LLM drafts the letter entirely
            data_summary = "No ERP data needed. Draft a formal leave application for the student."

        elif intent == "RAG":
            # Phase 2 placeholder
            data_summary = "Knowledge base search is coming in a future update."

        else:  # UNKNOWN
            data_summary = ""

    except Exception as exc:
        logger.error("Tool executor error", intent=intent, error=str(exc))
        data_summary = "I couldn't retrieve your data right now. Please try again in a moment."

    return {
        **state,
        "fetched_data": fetched,
        "data_summary": data_summary,
        "sources": sources,
    }
