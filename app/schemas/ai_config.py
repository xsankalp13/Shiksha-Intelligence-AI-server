"""
Pydantic schemas for the AI Config API.
These mirror what is stored in Redis and returned to the super-admin frontend.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal


class ModelOption(BaseModel):
    """
    Represents a single selectable model in the UI dropdown.
    """
    id: str
    display_name: str
    provider: str
    description: str
    context_window: int
    cost_input_per_1m: float
    cost_output_per_1m: float
    is_free: bool
    tags: list[str]


class AIConfig(BaseModel):
    """
    The active AI configuration stored in Redis.
    Superadmin can update this; all chat requests use it.
    """
    active_model: str = Field(..., description="Model ID from the registry, e.g. 'gemini-flash'")
    intent_classifier_model: str = Field(
        default="gemini-flash",
        description="Separate model for the cheap intent classification step"
    )
    max_conversation_turns: int = Field(
        default=6,
        ge=2, le=20,
        description="Number of recent turns to keep verbatim in context"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0, le=2.0,
        description="LLM temperature — lower = more deterministic"
    )
    max_output_tokens: int = Field(
        default=512,
        ge=64, le=4096,
        description="Maximum tokens the model can produce per response"
    )


class AIConfigUpdateRequest(BaseModel):
    active_model: str | None = None
    intent_classifier_model: str | None = None
    max_conversation_turns: int | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


class AIConfigResponse(BaseModel):
    config: AIConfig
    available_models: list[ModelOption]
