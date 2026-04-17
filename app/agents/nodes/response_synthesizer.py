"""
Node 3 — Response Synthesizer

Calls the main LLM (whichever model is active) with:
- Role-specific system prompt (STUDENT → warm; TEACHER → analytical)
- Compressed data summary (not raw JSON — huge token saving)
- RAG context block (when intent == RAG)
- Long-term student profile preferences (language, response length)
- Last N conversation turns (from Redis)
- The student's query

Outputs a clean, helpful, human response.
"""
from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.agents.state import AgentState
from app.core.logging import logger
from app.core.model_registry import build_llm

# ── System prompts ─────────────────────────────────────────────────────────────

_STUDENT_SYSTEM_PROMPT = """You are Shiksha AI, a friendly and knowledgeable student assistant for an Indian school ERP system.

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

_TEACHER_SYSTEM_PROMPT = """You are Shiksha AI, an analytical assistant for school teachers.

Your personality:
- Professional, data-first, and precise
- Use tables or bullet lists for class-level data
- Highlight students at risk clearly
- Use Indian currency symbol ₹ for financial data

Your rules:
- Only answer based on data provided in the context
- Never fabricate student information
- Be concise — teachers need quick actionable insights
- Keep responses under 200 words unless generating a class report
"""


def _get_system_prompt(role: str, profile: dict) -> str:
    """Select and optionally personalise the system prompt based on role and profile."""
    base = _TEACHER_SYSTEM_PROMPT if role in ("TEACHER", "ADMIN", "SUPER_ADMIN") else _STUDENT_SYSTEM_PROMPT

    # Personalise for language preference from long-term profile
    lang = profile.get("language_pref", "English")
    if lang != "English":
        base += f"\n\nIMPORTANT: Respond in {lang} (and English where needed for technical terms)."

    return base


async def response_synthesizer_node(state: AgentState) -> AgentState:
    query = state["query"]
    data_summary = state.get("data_summary", "")
    rag_context = state.get("rag_context", "")
    history = state.get("conversation_history", [])
    memory_summary = state.get("memory_summary", "")
    intent = state["intent"]
    role = state.get("role", "STUDENT")
    active_model = state.get("active_model", "gemini-flash")
    profile = state.get("student_profile", {})

    llm = build_llm(
        active_model,
        temperature=state.get("temperature", 0.3),
        max_tokens=state.get("max_output_tokens", 512),
    )

    # ── Build message list ────────────────────────────────────────────────
    system_prompt = _get_system_prompt(role, profile)
    messages: list = [SystemMessage(content=system_prompt)]

    # Inject compressed memory for context continuity
    if memory_summary:
        messages.append(
            HumanMessage(content=f"[Context from earlier in this session]\n{memory_summary}")
        )
        messages.append(AIMessage(content="Got it, I have context from our earlier conversation."))

    # Add recent verbatim turns
    for turn in history:
        r = turn.get("role", "user")
        content = turn.get("content", "")
        messages.append(HumanMessage(content=content) if r == "user" else AIMessage(content=content))

    # Inject ERP data context block
    if data_summary:
        context_block = f"[Current student data]\n{data_summary}"
        messages.append(HumanMessage(content=context_block))
        messages.append(AIMessage(content="I've reviewed your data. What would you like to know?"))

    # Inject RAG knowledge base context (Phase 2)
    if rag_context:
        messages.append(HumanMessage(
            content=(
                "[Knowledge Base — retrieved from school documents]\n"
                f"{rag_context}\n\n"
                "Use ONLY the above information to answer the student's question. "
                "If the answer isn't in the knowledge base, say so clearly."
            )
        ))
        messages.append(AIMessage(content="I've found relevant sections from the school's knowledge base."))

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
        role=role,
        response_chars=len(response_text),
    )
    return {**state, "response": response_text}
