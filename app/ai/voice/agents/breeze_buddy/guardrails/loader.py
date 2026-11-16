"""Load the agent's GUARDRAIL row per turn (no cache: changes apply on the
next message). A failed read runs the turn without guardrails."""

from typing import Any, Optional

from pydantic import ValidationError

from app.ai.voice.agents.breeze_buddy.accounts import accounts_for_template
from app.core.logger import logger
from app.database.accessor.breeze_buddy.evaluation_config import (
    get_evaluation_config,
)
from app.utils.common import parse_json

from .config import GuardrailsConfig
from .guard import TurnGuard
from .judge import LOG_COMPONENT

EVALUATION_TYPE = "GUARDRAIL"


async def load_turn_guard(
    *,
    template: Any,
    session: Any,
) -> Optional[TurnGuard]:
    """The turn's guard, or None when guardrails are off."""
    template_id = template.id
    try:
        row = await get_evaluation_config(str(template_id), EVALUATION_TYPE)
    except Exception:
        logger.bind(component=LOG_COMPONENT, alert="guardrail_config_read_failed").opt(
            exception=True
        ).error(
            f"guardrail config read failed for template {template_id}; "
            "running the turn without guardrails"
        )
        return None
    if not row or not row.get("enabled"):
        return None
    try:
        config = GuardrailsConfig.model_validate(parse_json(row, "configuration") or {})
    except ValidationError:
        # Unreadable row: log and skip, never break the chat.
        logger.bind(component=LOG_COMPONENT).opt(exception=True).error(
            f"unreadable guardrail config for template {template_id}; skipping"
        )
        return None
    if not config.active:
        return None
    channel = getattr(session, "current_channel", None)
    return TurnGuard(
        config,
        config_id=str(row["id"]),
        template_id=str(template_id),
        session_id=str(session.id),
        reseller_id=session.reseller_id,
        merchant_id=session.merchant_id,
        channel=getattr(channel, "value", None) or "CHAT",
        accounts=accounts_for_template(template),
        template_llm=getattr(
            getattr(template, "configurations", None), "llm_configurations", None
        ),
    )
