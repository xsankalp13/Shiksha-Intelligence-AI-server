"""
Node 1 — Intent Classifier

Token budget: < 80 tokens total (system + user).
Always uses the lightweight model (gemini-flash or gemini-flash-lite)
regardless of what the main active_model is.
Returns one of the 8 intent labels as a single JSON field.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.config import settings
from app.core.logging import logger
from app.core.model_registry import build_llm

# ── Static system prompt (does NOT change — ideal for caching) ────────────────
_SYSTEM_PROMPT = (
    "You are an intent classifier for a school ERP assistant. "
    "Classify the student query into EXACTLY ONE label. "
    "Respond with ONLY the label — no explanation, no punctuation.\n\n"
    "Labels:\n"
    "ATTENDANCE — attendance %, absent days, risk status\n"
    "FINANCE    — fees, invoices, payments, dues\n"
    "EXAM       — marks, results, grades, exam schedules\n"
    "SCHEDULE   — timetable, current class, next class\n"
    "GENERAL    — overview, profile, dashboard summary\n"
    "RAG        — syllabus, notes, chapters, concepts\n"
    "LEAVE_REQUEST — applying for leave, drafting excuse\n"
    "UNKNOWN    — unclear or unrelated"
)

async def intent_classifier_node(state: AgentState) -> AgentState:
    """Classify the query and write the intent back to state."""
    query = state["query"]
    model_id = state.get("intent_classifier_model", settings.INTENT_CLASSIFIER_MODEL)

    try:
        llm = build_llm(model_id, temperature=0.0, max_tokens=10)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        result = await llm.ainvoke(messages)
        raw = result.content.strip().upper()

        valid_intents = {
            "ATTENDANCE", "FINANCE", "EXAM", "SCHEDULE",
            "GENERAL", "RAG", "LEAVE_REQUEST", "UNKNOWN",
        }
        intent = raw if raw in valid_intents else "UNKNOWN"

    except Exception as exc:
        logger.warning("Intent classification failed, defaulting to GENERAL", error=str(exc))
        intent = "GENERAL"

    logger.info("Intent classified", query=query[:80], intent=intent)
    return {**state, "intent": intent}
