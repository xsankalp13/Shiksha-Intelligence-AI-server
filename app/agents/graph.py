"""
LangGraph StateGraph — wires all nodes together.

Flow:
  START → intent_classifier → tool_executor → response_synthesizer → guardrails → END

Conditional edges based on intent:
  UNKNOWN → skip tool_executor, go straight to synthesizer (ask for clarification)
  LEAVE_REQUEST → skip tool_executor entirely

The graph is compiled ONCE at module load and reused for all requests.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.nodes.intent_classifier import intent_classifier_node
from app.agents.nodes.tool_executor import tool_executor_node
from app.agents.nodes.response_synthesizer import response_synthesizer_node
from app.agents.nodes.guardrails import guardrails_node


def _should_fetch_data(state: AgentState) -> str:
    """
    Conditional edge: decide whether to hit the ERP tools.
    UNKNOWN and LEAVE_REQUEST don't need ERP data.
    """
    if state["intent"] in ("UNKNOWN", "LEAVE_REQUEST"):
        return "skip_tools"
    return "fetch_tools"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("response_synthesizer", response_synthesizer_node)
    graph.add_node("guardrails", guardrails_node)

    # ── Edges ─────────────────────────────────────────────────────────────
    graph.add_edge(START, "intent_classifier")

    graph.add_conditional_edges(
        "intent_classifier",
        _should_fetch_data,
        {
            "fetch_tools": "tool_executor",
            "skip_tools": "response_synthesizer",
        },
    )

    graph.add_edge("tool_executor", "response_synthesizer")
    graph.add_edge("response_synthesizer", "guardrails")
    graph.add_edge("guardrails", END)

    return graph.compile()


# Singleton — compiled once on import
shiksha_graph = build_graph()
