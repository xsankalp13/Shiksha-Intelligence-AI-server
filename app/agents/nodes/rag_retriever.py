"""
Node 2a — RAG Retriever

Runs ONLY when intent == "RAG". Queries the vector store (Pinecone or
ChromaDB) with the student's query and writes the retrieved text chunks
into state["rag_context"]. For all other intents this is a pass-through.

The response_synthesizer then injects rag_context into the LLM prompt
as a [Knowledge Base] section, grounding the answer in school documents.

Phase 2: Syllabus PDFs, handbooks, teacher notes, policies.
"""
from __future__ import annotations

from app.agents.state import AgentState
from app.core.logging import logger
from app.services.rag_service import RagService


async def rag_retriever_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant chunks from the knowledge base for RAG queries.
    Returns state unchanged (rag_context = "") for non-RAG intents.
    """
    intent = state["intent"]

    if intent != "RAG":
        return {**state, "rag_context": ""}

    query = state["query"]

    try:
        chunks = await RagService.query(query)

        if chunks:
            # Format chunks into a readable context block for the LLM
            rag_context = "\n\n---\n\n".join(
                f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)
            )
            sources = state.get("sources", []) + ["knowledge-base (RAG)"]
        else:
            rag_context = ""
            sources = state.get("sources", [])
            logger.warning("RAG retrieval returned no chunks", query=query[:60])

    except Exception as exc:
        logger.error("RAG retriever node failed", error=str(exc))
        rag_context = ""
        sources = state.get("sources", [])

    logger.info(
        "RAG retrieved",
        query=query[:60],
        chunks=len(chunks) if "chunks" in dir() else 0,
    )
    return {**state, "rag_context": rag_context, "sources": sources}
