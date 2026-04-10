"""
Model Registry — Single source of truth for all supported LLM models.

Adding a new model: just add an entry to MODEL_REGISTRY.
The graph, intent classifier, and config API all read from here.

Token cost estimates are per 1M tokens (input / output) in USD.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.language_models import BaseChatModel


class ModelProvider(str, Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    GROQ = "groq"
    OPENROUTER = "openrouter"


@dataclass
class ModelDefinition:
    """Describes a single available model."""
    id: str                          # Stable internal key (stored in Redis)
    display_name: str                # Shown in the UI dropdown
    provider: ModelProvider
    model_name: str                  # Actual model identifier for the SDK
    description: str
    context_window: int              # Max tokens (input + output)
    cost_input_per_1m: float         # USD per 1M input tokens (0 = free tier)
    cost_output_per_1m: float        # USD per 1M output tokens
    is_free: bool = False
    supports_system_prompt_caching: bool = False
    tags: list[str] = field(default_factory=list)


# ── Registry ─────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "gemini-3-flash": ModelDefinition(
        id="gemini-3-flash",
        display_name="Gemini 3 Flash (Google)",
        provider=ModelProvider.GOOGLE,
        model_name="gemini-3-flash-preview",
        description="Google's newest speed-optimized Gemini 3 Flash model.",
        context_window=1_000_000,
        cost_input_per_1m=0.075,
        cost_output_per_1m=0.30,
        supports_system_prompt_caching=True,
        tags=["recommended", "fast", "latest"],
    ),

    "gemini-flash": ModelDefinition(
        id="gemini-flash",
        display_name="Gemini 1.5 Flash (Google)",
        provider=ModelProvider.GOOGLE,
        model_name="gemini-1.5-flash-latest",
        description="Google's fastest, most cost-efficient model. Best default choice.",
        context_window=1_000_000,
        cost_input_per_1m=0.075,
        cost_output_per_1m=0.30,
        supports_system_prompt_caching=True,
        tags=["recommended", "fast", "low-cost"],
    ),
    "gemini-flash-lite": ModelDefinition(
        id="gemini-flash-lite",
        display_name="Gemini 1.5 Flash 8B (Google)",
        provider=ModelProvider.GOOGLE,
        model_name="gemini-1.5-flash-8b-latest",
        description="Ultra-cheap Gemini variant. Good for the intent classifier.",
        context_window=1_000_000,
        cost_input_per_1m=0.0375,
        cost_output_per_1m=0.15,
        tags=["ultra-cheap", "classifier"],
    ),
    "gemini-1.5-flash": ModelDefinition(
        id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash (Most Stable)",
        provider=ModelProvider.GOOGLE,
        model_name="gemini-1.5-flash",
        description="The most stable production-ready Flash model.",
        context_window=1_000_000,
        cost_input_per_1m=0.075,
        cost_output_per_1m=0.30,
        tags=["stable", "production"],
    ),

    # ── OpenAI ───────────────────────────────────────────────────────────
    "gpt-4o-mini": ModelDefinition(
        id="gpt-4o-mini",
        display_name="GPT-4o Mini (OpenAI)",
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        description="OpenAI's small, fast model. Great balance of quality and cost.",
        context_window=128_000,
        cost_input_per_1m=0.15,
        cost_output_per_1m=0.60,
        tags=["openai", "reliable"],
    ),

    # ── Groq ─────────────────────────────────────────────────────────────
    "groq-llama-3.3-70b": ModelDefinition(
        id="groq-llama-3.3-70b",
        display_name="LLaMA 3.3 70B (Groq)",
        provider=ModelProvider.GROQ,
        model_name="llama-3.3-70b-versatile",
        description="Meta's LLaMA 3 70B running on Groq's ultra-fast inference chips.",
        context_window=128_000,
        cost_input_per_1m=0.59,
        cost_output_per_1m=0.79,
        tags=["groq", "fast-inference", "large"],
    ),
    "groq-llama-3.1-8b": ModelDefinition(
        id="groq-llama-3.1-8b",
        display_name="LLaMA 3.1 8B Instant (Groq)",
        provider=ModelProvider.GROQ,
        model_name="llama-3.1-8b-instant",
        description="Tiny but blazing fast on Groq. Ideal for simple queries.",
        context_window=128_000,
        cost_input_per_1m=0.05,
        cost_output_per_1m=0.08,
        tags=["groq", "ultra-fast", "low-cost"],
    ),
    "groq-mixtral-8x7b": ModelDefinition(
        id="groq-mixtral-8x7b",
        display_name="Mixtral 8×7B (Groq)",
        provider=ModelProvider.GROQ,
        model_name="mixtral-8x7b-32768",
        description="Mistral's MoE model, strong at reasoning, on Groq hardware.",
        context_window=32_768,
        cost_input_per_1m=0.24,
        cost_output_per_1m=0.24,
        tags=["groq", "reasoning"],
    ),

    # ── OpenRouter — Free Models ──────────────────────────────────────────
    "openrouter-mistral-7b-free": ModelDefinition(
        id="openrouter-mistral-7b-free",
        display_name="Mistral 7B Instruct (OpenRouter Free)",
        provider=ModelProvider.OPENROUTER,
        model_name="mistralai/mistral-7b-instruct:free",
        description="Mistral 7B via OpenRouter's free tier. Rate-limited.",
        context_window=32_768,
        cost_input_per_1m=0.0,
        cost_output_per_1m=0.0,
        is_free=True,
        tags=["free", "openrouter"],
    ),
    "openrouter-gemma-7b-free": ModelDefinition(
        id="openrouter-gemma-7b-free",
        display_name="Gemma 7B (OpenRouter Free)",
        provider=ModelProvider.OPENROUTER,
        model_name="google/gemma-7b-it:free",
        description="Google's Gemma 7B via OpenRouter's free tier.",
        context_window=8_192,
        cost_input_per_1m=0.0,
        cost_output_per_1m=0.0,
        is_free=True,
        tags=["free", "openrouter"],
    ),
    "openrouter-llama-3-8b-free": ModelDefinition(
        id="openrouter-llama-3-8b-free",
        display_name="LLaMA 3 8B (OpenRouter Free)",
        provider=ModelProvider.OPENROUTER,
        model_name="meta-llama/llama-3-8b-instruct:free",
        description="Meta LLaMA 3 8B via OpenRouter free tier.",
        context_window=8_192,
        cost_input_per_1m=0.0,
        cost_output_per_1m=0.0,
        is_free=True,
        tags=["free", "openrouter"],
    ),
}

MODEL_OPTIONS_ORDERED: list[str] = [
    "gemini-3-flash",
    "gemini-flash",
    "gemini-flash-lite",
    "gemini-1.5-flash",
    "gpt-4o-mini",
    "groq-llama-3.3-70b",
    "groq-llama-3.1-8b",
    "groq-mixtral-8x7b",
    "openrouter-mistral-7b-free",
    "openrouter-gemma-7b-free",
    "openrouter-llama-3-8b-free",
]


def get_model_definition(model_id: str) -> ModelDefinition:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id '{model_id}'. Valid: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id]


def build_llm(model_id: str, **kwargs: Any) -> BaseChatModel:
    """
    Factory — returns a LangChain-compatible chat model for the given model_id.
    All extra kwargs (temperature, max_tokens, etc.) are forwarded.
    """
    from app.core.config import settings

    defn = get_model_definition(model_id)

    if defn.provider == ModelProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=defn.model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            **kwargs,
        )

    elif defn.provider == ModelProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=defn.model_name,
            openai_api_key=settings.OPENAI_API_KEY,
            **kwargs,
        )

    elif defn.provider == ModelProvider.GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=defn.model_name,
            groq_api_key=settings.GROQ_API_KEY,
            **kwargs,
        )

    elif defn.provider == ModelProvider.OPENROUTER:
        # OpenRouter is OpenAI-compatible — just swap the base URL & key
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=defn.model_name,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://shikshaintelligence.com",
                "X-Title": settings.OPENROUTER_APP_NAME,
            },
            **kwargs,
        )

    raise ValueError(f"Unsupported provider: {defn.provider}")
