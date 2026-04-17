"""
Node 1 — Intent Classifier

Token budget: < 80 tokens total (system + user).
Always uses the lightweight model (gemini-flash or gemini-flash-lite)
regardless of what the main active_model is.

Phase 2 additions:
- Role-aware routing: TEACHER gets different possible intents
- New intents: SUBMIT_LEAVE, TEACHER_CLASS, TEACHER_ATTENDANCE
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.config import settings
from app.core.logging import logger
from app.core.model_registry import build_llm

# ── System prompts ─────────────────────────────────────────────────────────────

_STUDENT_SYSTEM_PROMPT = (
    "You are an intent classifier for a school ERP assistant. "
    "Classify the student query into EXACTLY ONE label. "
    "Respond with ONLY the label — no explanation, no punctuation.\n\n"
    "Labels:\n"
    "ATTENDANCE    — attendance %, absent days, risk status\n"
    "FINANCE       — fees, invoices, payments, dues\n"
    "EXAM          — marks, results, grades, exam schedules\n"
    "SCHEDULE      — timetable, current class, next class\n"
    "GENERAL       — overview, profile, dashboard summary\n"
    "RAG           — syllabus, notes, chapters, concepts, policies\n"
    "LEAVE_REQUEST — composing/drafting a leave letter or excuse\n"
    "SUBMIT_LEAVE  — actually filing/submitting a leave application\n"
    "UNKNOWN       — unclear or unrelated"
)

_TEACHER_SYSTEM_PROMPT = (
    "You are an intent classifier for a school ERP assistant used by teachers. "
    "Classify the teacher query into EXACTLY ONE label. "
    "Respond with ONLY the label — no explanation, no punctuation.\n\n"
    "Labels:\n"
    "TEACHER_CLASS      — class performance, marks summary, subject averages\n"
    "TEACHER_ATTENDANCE — section attendance, at-risk students, absent counts\n"
    "LEAVE_REQUEST      — drafting communications or notices to parents\n"
    "RAG                — syllabus queries, policy questions, handbooks\n"
    "GENERAL            — general school overview, dashboard\n"
    "UNKNOWN            — unclear or unrelated"
)

_VALID_STUDENT_INTENTS = {
    "ATTENDANCE", "FINANCE", "EXAM", "SCHEDULE",
    "GENERAL", "RAG", "LEAVE_REQUEST", "SUBMIT_LEAVE", "UNKNOWN",
}

_VALID_TEACHER_INTENTS = {
    "TEACHER_CLASS", "TEACHER_ATTENDANCE",
    "LEAVE_REQUEST", "RAG", "GENERAL", "UNKNOWN",
}


async def intent_classifier_node(state: AgentState) -> AgentState:
    """Classify the query and write the intent back to state."""
    query = state["query"]
    role = state.get("role", "STUDENT")
    model_id = state.get("intent_classifier_model", settings.INTENT_CLASSIFIER_MODEL)

    # ── Select system prompt & valid labels based on role ─────────────────
    if role in ("TEACHER", "ADMIN", "SUPER_ADMIN"):
        system_prompt = _TEACHER_SYSTEM_PROMPT
        valid_intents = _VALID_TEACHER_INTENTS
    else:
        system_prompt = _STUDENT_SYSTEM_PROMPT
        valid_intents = _VALID_STUDENT_INTENTS

    try:
        llm = build_llm(model_id, temperature=0.0, max_tokens=10)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
        result = await llm.ainvoke(messages)
        raw = result.content.strip().upper()
        intent = raw if raw in valid_intents else "UNKNOWN"

    except Exception as exc:
        logger.warning("Intent classification failed, defaulting to GENERAL", error=str(exc))
        intent = "GENERAL"

    logger.info("Intent classified", query=query[:80], intent=intent, role=role)
    return {**state, "intent": intent}
