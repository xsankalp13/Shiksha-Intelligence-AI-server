"""
Node 1.5 — Entity Extractor

Runs ONLY when intent == "EXAM". Uses a fast micro-LLM call to pull
structured entities from the raw query so that tool_executor can make
targeted ERP calls (e.g. specific exam/subject) instead of falling
back to the generic dashboard.

Token budget: < 60 tokens. Output is always a compact JSON object.

Example:
    Input:  "What did I score in Term 1 Chemistry?"
    Output: {"exam_name": "Term 1", "subject": "Chemistry", "schedule_id": null}

For all other intents, this node is a pure pass-through (no LLM call).
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.config import settings
from app.core.logging import logger
from app.core.model_registry import build_llm


_SYSTEM_PROMPT = (
    "You are an entity extractor for a school ERP assistant. "
    "Extract structured information from the student query. "
    "Respond with ONLY a valid JSON object — no explanation, no markdown. "
    "Fields to extract (use null if not found):\n"
    '{"exam_name": string|null, "subject": string|null, '
    '"month": string|null, "year": string|null, "schedule_id": null}'
)


def _parse_entity_json(raw: str) -> dict:
    """Safely parse the LLM's JSON output. Falls back to empty dict on error."""
    try:
        # Strip markdown fences if the LLM added them despite instructions
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {}


async def entity_extractor_node(state: AgentState) -> AgentState:
    """
    Extract query entities for EXAM intent.
    For all other intents, returns state unchanged (zero overhead).
    """
    intent = state["intent"]

    # ── Fast pass-through for non-EXAM intents ────────────────────────────
    if intent != "EXAM":
        return {**state, "extracted_entities": {}}

    query = state["query"]
    model_id = state.get("intent_classifier_model", settings.INTENT_CLASSIFIER_MODEL)

    try:
        # Use the classifier model (lightweight) — capped at 80 output tokens
        llm = build_llm(model_id, temperature=0.0, max_tokens=80)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        result = await llm.ainvoke(messages)
        entities = _parse_entity_json(result.content)

    except Exception as exc:
        logger.warning("Entity extraction failed", error=str(exc))
        entities = {}

    logger.info("Entities extracted", intent=intent, entities=entities)
    return {**state, "extracted_entities": entities}
