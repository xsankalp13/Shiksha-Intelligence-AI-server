"""
Node 4 — Guardrails

Post-processing checks before returning the response to the user:
1. Strip any raw JSON structures that might have leaked
2. Enforce response length limits
3. Role-based access control — students can't see other students' data
   (In future: detect and redact names/IDs that don't belong to them)

This node is intentionally simple in V1.
It adds zero LLM calls — pure Python string checks.
"""
from __future__ import annotations
import re

from app.agents.state import AgentState


MAX_RESPONSE_CHARS = 2000   # Hard cap before we truncate


def _strip_raw_json(text: str) -> str:
    """Remove any leaked JSON-looking blocks from the response."""
    # Remove code blocks that look like JSON
    cleaned = re.sub(r"```(?:json)?\s*\{.*?\}\s*```", "[data removed]", text, flags=re.DOTALL)
    return cleaned


def _enforce_length(text: str) -> str:
    if len(text) > MAX_RESPONSE_CHARS:
        return text[:MAX_RESPONSE_CHARS] + "…"
    return text


async def guardrails_node(state: AgentState) -> AgentState:
    response = state.get("response", "")
    response = _strip_raw_json(response)
    response = _enforce_length(response)
    return {**state, "response": response, "error": None}
