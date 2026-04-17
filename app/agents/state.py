"""
AgentState — the shared TypedDict that flows through every LangGraph node.

Design notes:
- Role field enables future teacher/admin routing
- fetched_data stores raw tool results (summarizer compresses before LLM)
- sources tracks which ERP endpoints were called (returned to frontend)
- extracted_entities: structured data pulled by entity_extractor node
- rag_context: text chunks retrieved from the knowledge base
- student_profile: long-term preferences loaded from Supabase (optional)
"""
from __future__ import annotations

from typing import Any, TypedDict, Literal


class AgentState(TypedDict):
    # ── Identity ────────────────────────────────────────────────────────
    user_id: int
    role: Literal["STUDENT", "TEACHER", "ADMIN", "SUPER_ADMIN"]
    jwt_token: str                        # Forwarded to ERP tools

    # ── Request ─────────────────────────────────────────────────────────
    query: str
    session_id: str
    conversation_history: list[dict]      # [{role: user|assistant, content: ...}]
    memory_summary: str                   # Compressed older context

    # ── Processing ──────────────────────────────────────────────────────
    intent: str                           # Classified intent bucket
    extracted_entities: dict[str, Any]    # Phase 2: exam name, subject, dates, etc.
    fetched_data: dict[str, Any]          # Raw ERP JSON keyed by tool name
    data_summary: str                     # Compact text summary sent to LLM
    rag_context: str                      # Retrieved knowledge base chunks (RAG)

    # ── Model Config (resolved per request from Redis) ───────────────────
    active_model: str
    intent_classifier_model: str
    temperature: float
    max_output_tokens: int

    # ── Long-term Student Profile (optional, from Supabase) ───────────────
    student_profile: dict[str, Any]       # {language_pref, weak_subjects, ...}

    # ── Output ──────────────────────────────────────────────────────────
    response: str
    sources: list[str]                    # ERP endpoints called or RAG doc names
    error: str | None


# ── Intent Definitions ────────────────────────────────────────────────────────

INTENT_LABELS = Literal[
    "ATTENDANCE",       # Questions about attendance %
    "FINANCE",          # Fees, invoices, payments
    "EXAM",             # Marks, results, exam schedules
    "SCHEDULE",         # Today's timetable, next class
    "GENERAL",          # General ERP query — uses dashboard snapshot
    "RAG",              # Knowledge-base / notes / syllabus questions
    "LEAVE_REQUEST",    # Draft a leave application letter
    "SUBMIT_LEAVE",     # Actually file the leave in the ERP (action)
    "TEACHER_CLASS",    # Teacher querying class-level performance analytics
    "TEACHER_ATTENDANCE",  # Teacher checking section attendance risk
    "UNKNOWN",          # Fallback — ask for clarification
]
