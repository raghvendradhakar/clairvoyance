"""Per-agent evaluation config (the topics API pattern, type-generalized).

No row — or a disabled one — means the agent does not run the evaluation.
The admin configuration POST is the one creation point: it creates the
agent's row DISABLED or replaces the configuration of the existing one
(validated by the type's own validator); GET and the enable PATCH never
create and 404 without a row. The evaluation type travels in the body
(query string for GET), never in the path.

TOPIC is served here too (full-replace POST resolved through the topics
resolver, which fills its defaults); the /topics surface stays alongside
with its partial-patch semantics and the catalog add/remove.

GUARDRAIL (the live chat guardrail) follows template editing instead of
admin-only, and its save response carries preflight warnings.
"""

import copy
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

from app.ai.voice.agents.breeze_buddy.accounts import accounts_for_template
from app.ai.voice.agents.breeze_buddy.guardrails.config import GuardrailsConfig
from app.ai.voice.agents.breeze_buddy.guardrails.judge import (
    judge_account_problem,
    judge_answers,
)
from app.ai.voice.agents.breeze_buddy.guardrails.preflight import preflight_warnings
from app.ai.voice.agents.breeze_buddy.services.conversation_analysis.topics.extractor import (
    resolve_topic_evaluation_configuration,
)
from app.api.routers.breeze_buddy.templates.rbac import validate_template_access
from app.core.security.authorization import require_admin
from app.database.accessor.breeze_buddy.evaluation_config import (
    get_evaluation_config,
    save_evaluation_configuration,
    set_evaluation_enabled,
)
from app.database.accessor.breeze_buddy.template import get_template_by_id
from app.schemas import UserInfo
from app.schemas.breeze_buddy.auth import UserRole
from app.schemas.breeze_buddy.evals import (
    EvaluationConfigResponse,
    EvaluationEnableRequest,
    EvaluationType,
    SaveEvaluationConfigurationRequest,
)
from app.services.evals.definition import (
    validate_evals_configuration,
)
from app.utils.common import parse_json


def _config_response(
    template_id: str,
    evaluation_type: EvaluationType,
    row: Dict[str, Any],
    current_user: UserInfo,
    warnings: Optional[List[str]] = None,
) -> EvaluationConfigResponse:
    # The configuration (TOPIC's system prompt, an evals engine's rubrics) is
    # admin-only, as on /topics: everyone with template access sees the
    # enabled flag and the catalog, only admins the configuration itself.
    # GUARDRAIL is the merchant's own rules, so it follows template access.
    visible = (
        current_user.role == UserRole.ADMIN
        or evaluation_type is EvaluationType.GUARDRAIL
    )
    return EvaluationConfigResponse(
        template_id=template_id,
        evaluation_type=evaluation_type,
        enabled=bool(row.get("enabled")),
        # asyncpg hands jsonb back as text; parse_json takes either form
        configuration=(parse_json(row, "configuration") or {}) if visible else None,
        topics=list(row.get("topics") or []),
        warnings=warnings or [],
    )


async def _validate_access(template_id: str, current_user: UserInfo) -> Any:
    template = await get_template_by_id(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {template_id}",
        )
    validate_template_access(
        current_user,
        template.reseller_id,
        template.merchant_id,
        operation="manage evaluations for",
    )
    return template


async def get_evaluation_config_handler(
    template_id: str,
    evaluation_type: EvaluationType,
    current_user: UserInfo,
) -> EvaluationConfigResponse:
    await _validate_access(template_id, current_user)
    row = await get_evaluation_config(str(template_id), evaluation_type.value)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No {evaluation_type.value} configuration for this agent — "
                f"without one the evaluation does not run"
            ),
        )
    return _config_response(template_id, evaluation_type, row, current_user)


def _template_llm(template: Any) -> Any:
    return getattr(
        getattr(template, "configurations", None), "llm_configurations", None
    )


async def _refuse_silent_judge(
    template: Any, config: GuardrailsConfig, accounts: Any
) -> None:
    """Input checks fail closed: refuse to run them on a judge that does not
    answer, or every shopper message would be blocked."""
    if not config.input.enabled:
        return
    problem = await judge_answers(config.llm, accounts, _template_llm(template))
    if problem:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Guardrails cannot check shopper messages: {problem}",
        )


async def set_evaluation_enabled_handler(
    template_id: str,
    request: EvaluationEnableRequest,
    current_user: UserInfo,
) -> EvaluationConfigResponse:
    evaluation_type = request.evaluation_type
    template = await _validate_access(template_id, current_user)
    if evaluation_type is EvaluationType.GUARDRAIL and request.enabled:
        current = await get_evaluation_config(str(template_id), evaluation_type.value)
        if current:
            config = GuardrailsConfig.model_validate(
                parse_json(current, "configuration") or {}
            )
            await _refuse_silent_judge(
                template, config, accounts_for_template(template)
            )
    row = await set_evaluation_enabled(
        str(template_id), evaluation_type.value, request.enabled
    )
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No {evaluation_type.value} configuration for this agent — "
                f"without one the evaluation does not run"
            ),
        )
    return _config_response(template_id, evaluation_type, row, current_user)


def _validate_configuration(
    evaluation_type: EvaluationType,
    configuration: Any,
) -> Dict[str, Any]:
    """Each type validates its own shape. Raises ``ValueError``."""
    if evaluation_type is EvaluationType.CONVERSATION_EVALS:
        # decoded into the engine's own configuration type; the row keeps the
        # admin's own JSON, nothing defaulted into it
        validate_evals_configuration(configuration)
        return copy.deepcopy(configuration)
    if evaluation_type is EvaluationType.TOPIC:
        # the topics resolver: validates provider/sdk/model/region and fills
        # the runtime defaults, so the stored row is the resolved shape
        try:
            resolved = resolve_topic_evaluation_configuration(configuration)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        # the resolver tolerates a missing prompt (None); the runtime CHECK
        # does not — reject here so the caller gets a 400, not a 500
        if not resolved.get("system_prompt"):
            raise ValueError("system_prompt must be a non-empty string")
        return resolved
    if evaluation_type is EvaluationType.GUARDRAIL:
        # pydantic ValidationError is a ValueError
        return GuardrailsConfig.model_validate(configuration).model_dump(mode="json")
    raise ValueError(f"no validator for {evaluation_type.value}")


async def save_evaluation_configuration_handler(
    template_id: str,
    request: SaveEvaluationConfigurationRequest,
    current_user: UserInfo,
) -> EvaluationConfigResponse:
    evaluation_type = request.evaluation_type
    # GUARDRAIL needs template access only; other types stay admin-only.
    if evaluation_type is not EvaluationType.GUARDRAIL:
        require_admin(current_user)
    template = await _validate_access(template_id, current_user)

    try:
        normalized = _validate_configuration(evaluation_type, request.configuration)
    except (TypeError, ValueError) as exc:  # same net as /topics/configuration
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {evaluation_type.value} configuration: {exc}",
        ) from exc

    accounts = accounts_for_template(template)
    template_llm = _template_llm(template)
    guardrails: Optional[GuardrailsConfig] = None
    if evaluation_type is EvaluationType.GUARDRAIL:
        # Check the judge account before storing, like template save.
        guardrails = GuardrailsConfig.model_validate(normalized)
        problem = await judge_account_problem(guardrails.llm, accounts, template_llm)
        if problem:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid {evaluation_type.value} configuration: {problem}",
            )
        current = await get_evaluation_config(str(template_id), evaluation_type.value)
        if current and current.get("enabled"):  # live: must keep answering
            await _refuse_silent_judge(template, guardrails, accounts)

    # creates the row DISABLED when missing; enable is a separate flip
    row = await save_evaluation_configuration(
        str(template_id), evaluation_type.value, normalized
    )
    if not row:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not store {evaluation_type.value} configuration",
        )
    warnings = (
        await preflight_warnings(guardrails, accounts, template_llm)
        if guardrails is not None
        else None
    )
    return _config_response(template_id, evaluation_type, row, current_user, warnings)
