"""
Pydantic schemas for the Chat API.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The student's message")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID. If omitted, a new session is created and returned."
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    model_used: str
    sources: list[str] = Field(default_factory=list, description="ERP endpoints or RAG chunks used")
