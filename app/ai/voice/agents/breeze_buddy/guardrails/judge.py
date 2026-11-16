"""The guardrail judge: an observer-style LLM call. Like observers, it runs
on the template's own LLM (merge_llm_config + call_llm), with the guard's
optional ``llm`` override on top.

Measured 2026-11-16 (31 cases x 4 runs, Gemini API): gemini-3.6-flash, the
Assist template model, missed 0 of 12 attacks, p50 ~1.0 s, but 6 of 124
calls took over 2 s.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.ai.voice.agents.breeze_buddy.accounts import Accounts
from app.ai.voice.agents.breeze_buddy.llm import get_llm_service
from app.ai.voice.agents.breeze_buddy.observers.factory import merge_llm_config
from app.ai.voice.agents.breeze_buddy.observers.llm import call_llm
from app.ai.voice.llm.types import LLMConfiguration, LLMProvider, ThinkingConfiguration
from app.core.logger import logger

from .prompts import DECISION_TOOL, DECISION_TOOL_NAME, request_text, system_prompt

LOG_COMPONENT = "buddy.guardrails"
JUDGE_TIMEOUT_SECONDS = 2.0  # output: a timeout releases the text
# input fails closed, so it gets more time (gemini-3.6-flash: ~5% > 2 s)
JUDGE_INPUT_TIMEOUT_SECONDS = 3.5
# Also fill the fields an ``llm`` override leaves unset.
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 128
# Models that cannot turn thinking off spend output tokens on it.
JUDGE_MAX_TOKENS_THINKING = 1024
_OPENAI_REASONING = ("o1", "o3", "o4", "gpt-5")


@dataclass(frozen=True)
class JudgeResult:
    blocked: bool
    reason: str
    ms: int
    error: bool = False


def _is_openai_reasoning(cfg: LLMConfiguration) -> bool:
    """OpenAI/Azure reasoning models reject temperature and always think."""
    return cfg.provider in (None, LLMProvider.AZURE, LLMProvider.OPENAI) and (
        cfg.model or ""
    ).lower().startswith(_OPENAI_REASONING)


def _low_latency_thinking(cfg: LLMConfiguration) -> Optional[ThinkingConfiguration]:
    """The least thinking each model allows (pipecat's Gemini Flash defaults;
    call_llm does not apply them)."""
    name = (cfg.model or "").lower()
    if cfg.provider == LLMProvider.GOOGLE_VERTEX:
        if name.startswith("gemini-2.5-flash"):
            return ThinkingConfiguration(enabled=True, thinking_budget=0)
        if name.startswith("gemini-2.5"):  # Pro cannot turn thinking off
            return ThinkingConfiguration(enabled=True, thinking_budget=128)
        if name.startswith("gemini-3"):
            level = "minimal" if "flash" in name else "low"
            return ThinkingConfiguration(enabled=True, thinking_level=level)
    if _is_openai_reasoning(cfg):
        return ThinkingConfiguration(enabled=True, reasoning_effort="low")
    return None


def judge_llm_configuration(
    override: Optional[LLMConfiguration],
    template_llm: Optional[LLMConfiguration] = None,
) -> LLMConfiguration:
    """The guard's ``llm`` override merged onto the template's LLM, as for
    observers (``observers/factory.py``)."""
    base = template_llm or LLMConfiguration()
    judge_llm = override or LLMConfiguration()
    fill: Dict[str, Any] = {}
    if judge_llm.max_tokens is None:
        fill["max_tokens"] = JUDGE_MAX_TOKENS
    # Bedrock's GPT models reject the temperature field (see merge_llm_config).
    if judge_llm.temperature is None and judge_llm.provider != LLMProvider.AWS_BEDROCK:
        fill["temperature"] = JUDGE_TEMPERATURE
    merged = merge_llm_config(judge_llm.model_copy(update=fill), base)
    update: Dict[str, Any] = {}
    if merged.thinking is None:
        update["thinking"] = _low_latency_thinking(merged)
    if _is_openai_reasoning(merged) and judge_llm.temperature is None:
        # these reject temperature 0; use what the template's chat sends
        update["temperature"] = base.temperature
    name = (merged.model or "").lower()
    gemini_pro = (
        merged.provider == LLMProvider.GOOGLE_VERTEX
        and name.startswith("gemini")
        and "flash" not in name
    )
    if (_is_openai_reasoning(merged) or gemini_pro) and judge_llm.max_tokens is None:
        update["max_tokens"] = JUDGE_MAX_TOKENS_THINKING
    return merged.model_copy(update=update) if update else merged


async def build_judge_service(
    override: Optional[LLMConfiguration],
    accounts: Optional[Accounts] = None,
    template_llm: Optional[LLMConfiguration] = None,
) -> Optional[Any]:
    """The judge's service, or None if it cannot be built. ``accounts`` is
    the template's tenant, as for observers."""
    try:
        return await get_llm_service(
            judge_llm_configuration(override, template_llm),
            pooled=True,
            accounts=accounts,
        )
    except Exception as exc:
        logger.bind(component=LOG_COMPONENT).warning(
            f"guardrail judge service unavailable: error={exc!r}"
        )
        return None


async def judge_account_problem(
    override: Optional[LLMConfiguration],
    accounts: Accounts,
    template_llm: Optional[LLMConfiguration] = None,
) -> Optional[str]:
    """Why an override's ``credential_id`` is refused, or None. Same rule as
    template save."""
    if override is None or not override.credential_id:
        return None
    try:
        await accounts.get(judge_llm_configuration(override, template_llm))
    except ValueError as exc:  # AccountRefused subclasses ValueError
        return str(exc)
    return None


async def judge(
    service: Optional[Any],
    *,
    direction: str,
    merchant_rules: str,
    candidate: str,
    recent: List[str],
    tool_data: Optional[str] = None,
    timeout: float = JUDGE_TIMEOUT_SECONDS,
) -> JudgeResult:
    """One allow/block decision. Never raises: any failure is ``error=True``."""
    started = time.monotonic()
    try:
        if service is None:
            raise RuntimeError("judge service unavailable")
        name, args = await asyncio.wait_for(
            call_llm(
                llm_service=service,
                transcript_text=request_text(candidate, recent, tool_data),
                system_prompt=system_prompt(direction, merchant_rules),
                tools=[DECISION_TOOL],
                observer_name=f"guardrail:{direction}",
                require_tool=True,
            ),
            timeout=timeout,
        )
        ms = int((time.monotonic() - started) * 1000)
        decision = (args or {}).get("decision") if name == DECISION_TOOL_NAME else None
        if decision not in ("allow", "block"):
            raise ValueError(f"no decision in judge reply (tool={name!r})")
        return JudgeResult(
            blocked=decision == "block",
            reason=str((args or {}).get("reason_code") or "none"),
            ms=ms,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        ms = int((time.monotonic() - started) * 1000)
        logger.bind(component=LOG_COMPONENT).warning(
            f"guardrail judge failed: direction={direction} ms={ms} error={exc!r}"
        )
        reason = "judge_timeout" if isinstance(exc, TimeoutError) else "judge_error"
        return JudgeResult(blocked=False, reason=reason, ms=ms, error=True)


async def judge_answers(
    override: Optional[LLMConfiguration],
    accounts: Optional[Accounts],
    template_llm: Optional[LLMConfiguration],
) -> Optional[str]:
    """Why the judge cannot answer (one test call), or None. Input checks fail
    closed, so a judge that never answers would block every message."""
    service = await build_judge_service(override, accounts, template_llm)
    if service is None:
        return "the judge model could not be set up"
    result = await judge(
        service,
        direction="input",
        merchant_rules="",
        candidate="Hi",
        recent=[],
        timeout=JUDGE_INPUT_TIMEOUT_SECONDS,
    )
    return "the judge model did not answer" if result.error else None
