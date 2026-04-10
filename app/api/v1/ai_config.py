"""
GET  /v1/ai-config        → Returns current config + all available models
PATCH /v1/ai-config       → Update active model / params (super-admin only)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.core.model_registry import MODEL_REGISTRY, MODEL_OPTIONS_ORDERED, get_model_definition
from app.schemas.ai_config import (
    AIConfig, AIConfigUpdateRequest, AIConfigResponse, ModelOption,
)
from app.services.session_service import SessionService

router = APIRouter()


def _registry_to_options() -> list[ModelOption]:
    return [
        ModelOption(
            id=mid,
            display_name=MODEL_REGISTRY[mid].display_name,
            provider=MODEL_REGISTRY[mid].provider.value,
            description=MODEL_REGISTRY[mid].description,
            context_window=MODEL_REGISTRY[mid].context_window,
            cost_input_per_1m=MODEL_REGISTRY[mid].cost_input_per_1m,
            cost_output_per_1m=MODEL_REGISTRY[mid].cost_output_per_1m,
            is_free=MODEL_REGISTRY[mid].is_free,
            tags=MODEL_REGISTRY[mid].tags,
        )
        for mid in MODEL_OPTIONS_ORDERED
    ]


async def _load_or_default_config() -> AIConfig:
    raw = await SessionService.get_ai_config()
    if raw:
        return AIConfig(**raw)
    return AIConfig(
        active_model=settings.ACTIVE_MODEL,
        intent_classifier_model=settings.INTENT_CLASSIFIER_MODEL,
    )


@router.get("/ai-config", response_model=AIConfigResponse)
async def get_ai_config():
    """Returns the current AI config and all available models for the dropdown."""
    config = await _load_or_default_config()
    return AIConfigResponse(
        config=config,
        available_models=_registry_to_options(),
    )


@router.patch("/ai-config", response_model=AIConfigResponse)
async def update_ai_config(update: AIConfigUpdateRequest):
    """
    Update AI configuration fields.
    In production, add a dependency here to check SUPER_ADMIN role from JWT.
    """
    config = await _load_or_default_config()

    # Validate model ID if provided
    if update.active_model:
        try:
            get_model_definition(update.active_model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # Merge updates
    updated = config.model_copy(
        update={k: v for k, v in update.model_dump().items() if v is not None}
    )

    # Persist to Redis
    await SessionService.set_ai_config(updated.model_dump())

    return AIConfigResponse(
        config=updated,
        available_models=_registry_to_options(),
    )
