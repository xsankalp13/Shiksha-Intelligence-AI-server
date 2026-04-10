"""
POST /v1/chat — Main chat endpoint.

Flow:
1. Decode JWT → extract user_id, role
2. Load/create Redis session → conversation history
3. Load active AI config from Redis (falls back to .env defaults)
4. Invoke the LangGraph
5. Persist the new turn to Redis
6. Return structured response
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app.agents.graph import shiksha_graph
from app.agents.state import AgentState
from app.core.config import settings
from app.core.logging import logger
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.ai_config import AIConfig
from app.services.session_service import SessionService

router = APIRouter()


def _decode_jwt(token: str) -> dict:
    """Decode the JWT and return claims. Raises 401 on failure."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=["HS256", "HS384", "HS512"],
            options={"verify_aud": False},
        )
        return payload
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")


def _extract_role(claims: dict) -> str:
    """Extract role from JWT claims. Defaults to STUDENT."""
    roles: list = claims.get("roles", claims.get("authorities", []))
    for r in roles:
        r_upper = r.upper().replace("ROLE_", "")
        if r_upper in ("SUPER_ADMIN", "ADMIN", "TEACHER", "STUDENT"):
            return r_upper
    return "STUDENT"


security = HTTPBearer(auto_error=False)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: HTTPAuthorizationCredentials | None = Depends(security),
):
    # ── Auth / Mock Logic ───────────────────────────────────────────────────
    user_id = 0
    role = "STUDENT"
    raw_token = ""

    if token:
        raw_token = token.credentials
        try:
            claims = _decode_jwt(raw_token)
            user_id = int(claims.get("user_id", claims.get("userId", 0)))
            role = _extract_role(claims)
        except HTTPException:
            if not settings.JWT_ALLOW_MOCK:
                raise
            # In mock mode, we gracefully ignore the error
            user_id = 999
            role = "STUDENT"
            logger.warning("Invalid JWT, using mock user 999 because JWT_ALLOW_MOCK=True")
    else:
        # No token provided in Swagger/Request
        if not settings.JWT_ALLOW_MOCK:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated, and mock mode is off."
            )
        user_id = 999
        role = "STUDENT"
        raw_token = "mock-dev-token"
        logger.info("No JWT provided, using mock user 999 because JWT_ALLOW_MOCK=True")

    logger.info("Chat request", user_id=user_id, role=role, query=request.query[:80])

    # ── Session ───────────────────────────────────────────────────────────
    session_id = request.session_id or SessionService.new_session_id()
    history_data = await SessionService.get_history(session_id)

    # ── AI Config (from Redis, fallback to defaults) ───────────────────────
    raw_config = await SessionService.get_ai_config()
    if raw_config:
        config = AIConfig(**raw_config)
    else:
        config = AIConfig(
            active_model=settings.ACTIVE_MODEL,
            intent_classifier_model=settings.INTENT_CLASSIFIER_MODEL,
        )

    # ── Build initial state ───────────────────────────────────────────────
    initial_state: AgentState = {
        "user_id": user_id,
        "role": role,
        "jwt_token": raw_token,
        "query": request.query,
        "session_id": session_id,
        "conversation_history": history_data["turns"],
        "memory_summary": history_data.get("summary", ""),
        "intent": "",
        "fetched_data": {},
        "data_summary": "",
        "active_model": config.active_model,
        "intent_classifier_model": config.intent_classifier_model,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "response": "",
        "sources": [],
        "error": None,
    }

    # ── Run the graph ─────────────────────────────────────────────────────
    final_state: AgentState = await shiksha_graph.ainvoke(initial_state)

    # ── Persist conversation turn ─────────────────────────────────────────
    await SessionService.append_turn(session_id, "user", request.query)
    await SessionService.append_turn(session_id, "assistant", final_state["response"])

    return ChatResponse(
        session_id=session_id,
        response=final_state["response"],
        intent=final_state["intent"],
        model_used=config.active_model,
        sources=final_state["sources"],
    )
