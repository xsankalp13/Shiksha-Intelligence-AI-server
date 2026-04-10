"""
Node 3 — Response Synthesizer

Calls the main LLM (whichever model is active) with:
- Static system prompt (cached where supported)
- Compressed data summary (not raw JSON — huge token saving)
- Last N conversation turns (from Redis)
- The student's query

Outputs a clean, helpful, human response.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.agents.state import AgentState
from app.core.logging import logger
from app.core.model_registry import build_llm

# ── Static system prompt ────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are Shiksha AI, a friendly and knowledgeable student assistant for an Indian school ERP system.

Your personality:
- Warm, encouraging, and supportive
- Concise but complete — never sprawling walls of text
- Use bullet points for lists, plain prose for conversational replies
- Use Indian currency symbol ₹ for monetary amounts
- When attendance is CRITICAL or WARNING, be gently motivating

Your rules:
- Only answer based on data provided in the context below
- Never fabricate marks, fees, or dates
- If the data is empty or missing, say so honestly and offer to help differently
- Do NOT expose raw JSON or technical field names
- Keep responses under 150 words unless it's a leave letter draft
"""


async def response_synthesizer_node(state: AgentState) -> AgentState:
    query = state["query"]
    data_summary = state.get("data_summary", "")
    history = state.get("conversation_history", [])
    memory_summary = state.get("memory_summary", "")
    intent = state["intent"]
    active_model = state.get("active_model", "gemini-flash")

    llm = build_llm(
        active_model,
        temperature=state.get("temperature", 0.3),
        max_tokens=state.get("max_output_tokens", 512),
    )

    # ── Build message list ────────────────────────────────────────────────
    messages: list = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Inject compressed memory for context continuity
    if memory_summary:
        messages.append(
            HumanMessage(content=f"[Context from earlier in this session]\n{memory_summary}")
        )
        messages.append(AIMessage(content="Got it, I have context from our earlier conversation."))

    # Add recent verbatim turns
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))

    # Add data context block (compact summary, not raw JSON)
    if data_summary:
        context_block = f"[Current student data]\n{data_summary}"
        messages.append(HumanMessage(content=context_block))
        messages.append(AIMessage(content="I've reviewed your data. What would you like to know?"))

    # The actual user query
    messages.append(HumanMessage(content=query))

    # ── LLM call ─────────────────────────────────────────────────────────
    try:
        result = await llm.ainvoke(messages)
        response_text = result.content.strip()
    except Exception as exc:
        logger.error("LLM call failed", model=active_model, error=str(exc))
        response_text = (
            "I encountered a temporary issue generating a response. "
            "Please try again in a moment."
        )

    logger.info(
        "Response generated",
        intent=intent,
        model=active_model,
        response_chars=len(response_text),
    )
    return {**state, "response": response_text}
