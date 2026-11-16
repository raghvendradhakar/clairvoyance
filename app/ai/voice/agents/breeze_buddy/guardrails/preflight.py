"""Save-time test: run the real checks on normal shopping texts and warn
about any the new rules would block."""

import asyncio
from typing import List, Optional, Tuple

from app.ai.voice.agents.breeze_buddy.accounts import Accounts
from app.ai.voice.llm.types import LLMConfiguration

from . import rules
from .config import GuardrailsConfig
from .judge import build_judge_service, judge

_SHOPPER_MESSAGES = [
    "Hi",
    "Do you have this in medium?",
    "What is the price of this one?",
    "Do you have any discount or sale right now?",
    "Where is my order 1234?",
    "How do I return an item?",
    "Do you deliver to Delhi?",
    "Mera order kab aayega?",
]

_TOOL_DATA = (
    '{"order": {"number": "1234", "status": "shipped"}, "sizes": ["S", "M", "L"]}'
)
_ASSISTANT_REPLIES = [
    "Hi! How can I help you shop today?",
    "Sorry, I can't help with that, but I can help you find a product.",
    "Your order #1234 has shipped.",
    "This one comes in S, M and L.",
    "You can see our current offers in the Sale collection.",
]


async def _blocked(
    service, direction: str, merchant_rules: str, text: str
) -> Tuple[bool, bool]:
    """(blocked, check_failed) for one sample."""
    regex_reason = (
        rules.check_input(text) if direction == "input" else rules.check_output(text)
    )
    if regex_reason:
        return True, False
    result = await judge(
        service,
        direction=direction,
        merchant_rules=merchant_rules,
        candidate=text,
        recent=[],
        tool_data=_TOOL_DATA if direction == "output" else None,
    )
    return result.blocked, result.error


async def preflight_warnings(
    config: GuardrailsConfig,
    accounts: Optional[Accounts] = None,
    template_llm: Optional[LLMConfiguration] = None,
) -> List[str]:
    samples = []
    if config.input.enabled:
        samples += [("input", config.input.rules, t) for t in _SHOPPER_MESSAGES]
    if config.output.enabled:
        samples += [("output", config.output.rules, t) for t in _ASSISTANT_REPLIES]
    if not samples:
        return []
    service = await build_judge_service(config.llm, accounts, template_llm)
    results = await asyncio.gather(*(_blocked(service, *s) for s in samples))
    warnings: List[str] = []
    for (direction, _, text), (blocked, failed) in zip(samples, results):
        if failed:
            warnings.append(f"Preflight could not check: {text!r}")
        elif blocked:
            who = "shopper message" if direction == "input" else "assistant reply"
            warnings.append(f"These rules would block a normal {who}: {text!r}")
    return warnings
