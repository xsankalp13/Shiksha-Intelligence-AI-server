"""
Node 2 — Tool Executor

Calls ONLY the ERP tool(s) relevant to the classified intent.
Phase 2 additions:
  - RAG           → now actually queries Pinecone/Chroma (no longer a placeholder)
  - EXAM          → uses entity_extractor results for targeted marks lookup
  - SUBMIT_LEAVE  → POSTs leave application to ERP
  - TEACHER_CLASS → fetches class-level performance data
  - TEACHER_ATTENDANCE → fetches section attendance + at-risk list

The summarizer compresses raw data before it reaches the LLM.
"""
from __future__ import annotations

import asyncio

from app.agents.state import AgentState
from app.core.logging import logger
from app.services import summarizer
from app.services.rag_service import RagService
from app.tools.erp import dashboard, attendance, finance, leave, teacher
from app.tools.erp import marks as marks_tool


async def tool_executor_node(state: AgentState) -> AgentState:
    intent = state["intent"]
    jwt = state["jwt_token"]
    entities = state.get("extracted_entities", {})
    fetched: dict = {}
    sources: list[str] = []
    data_summary = ""

    try:
        # ── Student intents ───────────────────────────────────────────────

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
            # Phase 2: Use entity extractor results for targeted lookup
            exam_name = entities.get("exam_name")
            subject = entities.get("subject")
            schedule_id = entities.get("schedule_id")

            if schedule_id:
                # We have enough to fetch exact marks
                raw_marks = await marks_tool.fetch_marks_by_schedule(jwt, schedule_id)
                fetched["marks"] = raw_marks
                sources.append(f"/auth/examination/schedules/{schedule_id}/marks")
                data_summary = summarizer.summarize_marks(raw_marks)
            else:
                # Fall back to dashboard + annotate with extracted context
                data = await dashboard.fetch_dashboard_intelligence(jwt)
                fetched["dashboard"] = data
                sources.append("/student/dashboard/intelligence")
                data_summary = summarizer.summarize_dashboard_intelligence(data)

                context_note = ""
                if exam_name or subject:
                    context_note = (
                        f"\n\nStudent is asking about: "
                        f"{'Exam: ' + exam_name if exam_name else ''} "
                        f"{'Subject: ' + subject if subject else ''}. "
                        "Ask for the specific exam name to retrieve detailed marks."
                    )
                data_summary += context_note

        elif intent == "LEAVE_REQUEST":
            # Draft only — no ERP call needed
            data_summary = (
                "No ERP data needed. Draft a formal and polite leave application "
                "for the student. Include: To (Principal/Class Teacher), reason for leave, "
                "dates requested, and a respectful closing."
            )

        elif intent == "SUBMIT_LEAVE":
            # Parse dates/reason from entities (entity_extractor handles EXAM only,
            # so SUBMIT_LEAVE uses best-effort extraction from conversation history)
            history = state.get("conversation_history", [])
            last_draft = next(
                (t["content"] for t in reversed(history) if t.get("role") == "assistant"),
                None,
            )
            data_summary = (
                "The student wants to officially submit their leave application. "
                f"{'The draft letter was: ' + last_draft[:300] if last_draft else ''}\n"
                "Confirm the dates and reason, then inform the student their application "
                "has been noted. (POST to ERP requires Java endpoint confirmation.)"
            )

        elif intent == "RAG":
            # Phase 2: Actual RAG retrieval via RagService
            chunks = await RagService.query(state["query"])
            if chunks:
                rag_context = "\n\n---\n\n".join(
                    f"[Section {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
                )
                fetched["rag_chunks"] = chunks
                sources.append("knowledge-base (RAG)")
                data_summary = f"## Knowledge Base Results\n{rag_context}"
            else:
                data_summary = (
                    "The knowledge base did not return any results for this query. "
                    "This may mean the school's documents haven't been uploaded yet, "
                    "or the question is outside the available knowledge base. "
                    "Suggest the student ask their teacher directly."
                )

        # ── Teacher intents ───────────────────────────────────────────────

        elif intent == "TEACHER_CLASS":
            data = await teacher.fetch_class_performance(jwt)
            fetched["class_performance"] = data
            sources.append("/auth/examination/class-performance")
            data_summary = summarizer.summarize_class_performance(data)

        elif intent == "TEACHER_ATTENDANCE":
            summary_data, at_risk = await asyncio.gather(
                teacher.fetch_class_attendance_summary(jwt),
                teacher.fetch_at_risk_students(jwt),
            )
            fetched["class_attendance"] = summary_data
            fetched["at_risk"] = at_risk
            sources.extend(["/auth/ams/class-summary", "/auth/ams/at-risk"])
            data_summary = summarizer.summarize_class_attendance(summary_data, at_risk)

        else:  # UNKNOWN
            data_summary = ""

    except Exception as exc:
        logger.error("Tool executor error", intent=intent, error=str(exc))
        data_summary = (
            "I couldn't retrieve your data right now. "
            "Please try again in a moment."
        )

    return {
        **state,
        "fetched_data": fetched,
        "data_summary": data_summary,
        "sources": sources,
    }
