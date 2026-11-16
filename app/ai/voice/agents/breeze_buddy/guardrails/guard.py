"""One chat turn's guardrails (sequential mode).

- Input: the full message is checked before the LLM runs. Judge error: block.
- Output: each LLM call's text is held and checked at ``COMMIT_POINT``,
  before rows are saved or tools run. Judge error: release.
- render_ui ops: regex only; a hit drops that op.
- One evaluation_result row per turn.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from app.ai.voice.agents.breeze_buddy.accounts import Accounts
from app.ai.voice.agents.breeze_buddy.chat.history.compactor import project_tool_json
from app.ai.voice.agents.breeze_buddy.chat.sse import SSEEvent
from app.ai.voice.llm.types import LLMConfiguration
from app.core.logger import logger
from app.database.accessor.breeze_buddy.evaluation_result import (
    save_evaluation_results,
)

from . import rules
from .config import GuardrailsConfig
from .judge import (
    JUDGE_INPUT_TIMEOUT_SECONDS,
    JUDGE_TIMEOUT_SECONDS,
    LOG_COMPONENT,
    build_judge_service,
    judge,
)

# Yielded by the cycle loop before it saves rows or runs tools (guard only).
COMMIT_POINT = "_guardrail_commit_point"

# Longer held text is checked before the LLM call ends.
EARLY_CHECK_CHARS = 1000
# Projected, URL-free tool data: local p50 4.4k for a search, ~3k for a cart.
TOOL_DATA_MAX_CHARS = 8000
RECENT_MESSAGES = 4
RECENT_MESSAGE_MAX_CHARS = 500

_pending_result_writes: "set[asyncio.Task[None]]" = set()


@dataclass
class Verdict:
    blocked: bool
    layer: str  # "regex" | "judge" | "none"
    reason: str = "none"
    judge_ms: Optional[int] = None
    judge_error: bool = False


@dataclass
class GateStep:
    """What the agent does after feeding one event to the output gate."""

    release: List[SSEEvent] = field(default_factory=list)
    blocked: Optional[Verdict] = None


def recent_messages(history: List[Dict[str, Any]]) -> List[str]:
    """The last few shopper/assistant text messages, for the judge's context."""
    out: List[str] = []
    for msg in history:
        role, content = msg.get("role"), msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            out.append(f"{role}: {content[:RECENT_MESSAGE_MAX_CHARS]}")
    return out[-RECENT_MESSAGES:]


def _without_urls(value: Any) -> Any:
    """Drop URL and gid strings (images, links): no facts for the judge."""
    if isinstance(value, str):
        return None if value.startswith(("http://", "https://", "gid://")) else value
    if isinstance(value, dict):
        kept = {k: _without_urls(v) for k, v in value.items()}
        return {k: v for k, v in kept.items() if v not in (None, {}, [])}
    if isinstance(value, list):
        kept_items = [_without_urls(v) for v in value]
        return [v for v in kept_items if v not in (None, {}, [])]
    return value


def turn_tool_data(
    messages: List[Any],
    projection: Optional[Dict[str, List[str]]],
    turn_start: int,
    max_chars: int = TOOL_DATA_MAX_CHARS,
) -> Optional[str]:
    """Tool results added to the context from ``turn_start`` (this turn),
    projected with the template's keep-lists (compactor.py), URLs dropped,
    newest first within ``max_chars``."""
    names: Dict[str, str] = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call, dict) else None
                if isinstance(fn, dict):
                    names[call.get("id", "")] = fn.get("name", "?")
    parts: List[str] = []
    for m in messages[turn_start:]:
        if not (isinstance(m, dict) and m.get("role") == "tool"):
            continue
        name = names.get(m.get("tool_call_id") or "", "tool")
        content = m.get("content")
        data: Any = project_tool_json(content, (projection or {}).get(name, []))
        if data is None:
            try:
                data = json.loads(content) if isinstance(content, str) else content
            except (TypeError, ValueError):
                data = content
        text = json.dumps(
            _without_urls(data), ensure_ascii=False, separators=(",", ":")
        )
        parts.append(f"{name}: {text}")
    kept: List[str] = []
    used = 0
    for part in reversed(parts):  # newest first
        if used >= max_chars:
            break
        kept.append(part[: max_chars - used])
        used += len(kept[-1]) + 1
    return "\n".join(reversed(kept)) or None


def ui_strings(value: Any) -> List[str]:
    """Every string inside a UI op (labels, titles, ids, URLs)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in ui_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in ui_strings(v)]
    return []


def visible_strings(value: Any) -> List[str]:
    """Readable strings in a UI op. Skips ids and URLs, which can look like
    card numbers."""
    return [
        text
        for text in (s.strip() for s in ui_strings(value))
        if " " in text and not text.startswith(("http://", "https://", "gid://"))
    ]


def ui_op_reason(op: Any) -> Optional[str]:
    """Reason code when a UI op must be dropped, else None."""
    if rules.has_secret("\n".join(ui_strings(op))):
        return rules.SECRET
    if rules.has_card_number("\n".join(visible_strings(op))):
        return rules.PAYMENT_CARD
    return None


class TurnGuard:
    def __init__(
        self,
        config: GuardrailsConfig,
        *,
        config_id: str,
        template_id: str,
        session_id: str,
        reseller_id: str,
        merchant_id: Optional[str],
        channel: str,
        accounts: Optional[Accounts] = None,
        template_llm: Optional[LLMConfiguration] = None,
    ) -> None:
        self.config = config
        self._accounts = accounts
        self._template_llm = template_llm
        self._config_id = config_id
        self._template_id = template_id
        self._session_id = session_id
        self._reseller_id = reseller_id
        self._merchant_id = merchant_id
        self._channel = channel
        self._service: Any = None
        self._started_at = datetime.now(timezone.utc)
        self._history: List[str] = []
        self._user_text = ""
        self.input_verdict: Optional[Verdict] = None
        self.output_verdicts: List[Verdict] = []
        self.ui_regex_hits = 0
        self.outcome = "allowed"

    @property
    def input_enabled(self) -> bool:
        return self.config.input.enabled

    @property
    def output_enabled(self) -> bool:
        return self.config.output.enabled

    def begin_turn(self, history: List[Dict[str, Any]], user_text: str) -> None:
        """Set the recent chat and the shopper message for this turn's checks."""
        self._history = recent_messages(history)
        self._user_text = user_text[:RECENT_MESSAGE_MAX_CHARS]

    async def _judge(self, direction: str, text: str, tool_data: Optional[str]):
        if self._service is None:
            self._service = await build_judge_service(
                self.config.llm, self._accounts, self._template_llm
            )
        if direction == "input":
            rule, recent = self.config.input, self._history
        else:
            rule = self.config.output
            recent = self._history + (
                [f"user: {self._user_text}"] if self._user_text else []
            )
        return await judge(
            self._service,
            direction=direction,
            merchant_rules=rule.rules,
            candidate=text,
            recent=recent[-RECENT_MESSAGES:],
            tool_data=tool_data,
            timeout=(
                JUDGE_INPUT_TIMEOUT_SECONDS
                if direction == "input"
                else JUDGE_TIMEOUT_SECONDS
            ),
        )

    def _alert_judge_error(self, direction: str) -> None:
        # Alert line, like the credits alert in chat/agent/core.py.
        logger.bind(
            component=LOG_COMPONENT,
            alert="guardrail_judge_error",
            direction=direction,
            merchant_id=self._merchant_id,
            template_id=self._template_id,
            session_id=self._session_id,
        ).warning(
            f"[GUARDRAIL_JUDGE_ERROR] {direction} check failed; "
            f"{'message blocked' if direction == 'input' else 'text released'}"
        )

    def ui_op_blocked(self, op: Any) -> bool:
        """True when this render_ui op must be dropped (regex hit)."""
        if ui_op_reason(op) is None:
            return False
        self.ui_regex_hits += 1
        return True

    async def check_input(self, text: str) -> Verdict:
        reason = rules.check_input(text)
        if reason:
            verdict = Verdict(blocked=True, layer="regex", reason=reason)
        else:
            result = await self._judge("input", text, None)
            if result.error:
                self._alert_judge_error("input")
            # fail closed: an unanswered check blocks the message
            verdict = Verdict(
                blocked=result.blocked or result.error,
                layer="judge",
                reason=result.reason,
                judge_ms=result.ms,
                judge_error=result.error,
            )
        self.input_verdict = verdict
        if verdict.blocked:
            self.outcome = "blocked_input"
        return verdict

    async def check_output(self, text: str, tool_data: Optional[str]) -> Verdict:
        reason = rules.check_output(text)
        if reason:
            verdict = Verdict(blocked=True, layer="regex", reason=reason)
        else:
            result = await self._judge("output", text, tool_data)
            if result.error:
                self._alert_judge_error("output")
            # fail open: an unanswered check releases the text
            verdict = Verdict(
                blocked=result.blocked and not result.error,
                layer="judge",
                reason=result.reason,
                judge_ms=result.ms,
                judge_error=result.error,
            )
        self.output_verdicts.append(verdict)
        if verdict.blocked:
            self.outcome = "blocked_output"
        return verdict

    def output_gate(
        self, tool_data: Optional[Callable[[], Optional[str]]] = None
    ) -> "FullTextGate":
        return FullTextGate(self, tool_data)

    # -- result row ---------------------------------------------------------

    def _metadata(self, turn_id: str) -> Dict[str, Any]:
        inp = self.input_verdict
        outs = self.output_verdicts
        blocked_out = next((v for v in outs if v.blocked), None)
        judge_times = [v.judge_ms for v in outs if v.judge_ms is not None]
        result_key = f"turn:{turn_id}"
        return {
            "type": result_key,
            "execution": self.config.execution,
            "channel": self._channel,
            "outcome": self.outcome,
            "input_decision": (
                "skipped" if inp is None else ("block" if inp.blocked else "allow")
            ),
            "input_layer": inp.layer if inp else None,
            "input_reason": inp.reason if inp else None,
            "input_judge_ms": inp.judge_ms if inp else None,
            "output_checks": len(outs),
            "output_decision": (
                "skipped" if not outs else ("block" if blocked_out else "allow")
            ),
            "output_layer": blocked_out.layer if blocked_out else None,
            "output_reason": blocked_out.reason if blocked_out else None,
            "output_judge_ms_max": max(judge_times) if judge_times else None,
            "ui_regex_hits": self.ui_regex_hits,
            "judge_errors": sum(
                1 for v in [inp, *outs] if v is not None and v.judge_error
            ),
        }

    def schedule_result_write(self, turn_id: str) -> None:
        """Write the turn's verdicts in a held background task, like billing."""
        if self.input_verdict is None and not self.output_verdicts:
            return
        metadata = self._metadata(turn_id)

        async def _write() -> None:
            try:
                await save_evaluation_results(
                    evaluation_config_id=self._config_id,
                    evaluation_type="GUARDRAIL",
                    source_id=self._session_id,
                    reseller_id=self._reseller_id,
                    merchant_id=self._merchant_id,
                    template_id=self._template_id,
                    started_at=self._started_at,
                    results=[metadata],
                )
            except Exception:
                logger.bind(component=LOG_COMPONENT).opt(exception=True).error(
                    f"guardrail result write failed session={self._session_id}"
                )

        task = asyncio.create_task(_write(), name=f"guardrail-result:{turn_id}")
        _pending_result_writes.add(task)
        task.add_done_callback(_pending_result_writes.discard)


class FullTextGate:
    """Holds an LLM call's events until its text is checked, at
    ``COMMIT_POINT``, turn end, or ``EARLY_CHECK_CHARS``."""

    def __init__(
        self,
        guard: TurnGuard,
        tool_data: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._guard = guard
        # Read at check time: the tool results the LLM has seen so far.
        self._tool_data = tool_data or (lambda: None)
        self._held: List[SSEEvent] = []
        self._text: List[str] = []
        # Shown but not yet saved; saved with the fixed reply on a block.
        self.released_unsaved = ""

    def _held_text(self) -> str:
        return "".join(self._text)

    async def _check_held(self) -> GateStep:
        text = self._held_text()
        if text.strip():
            verdict = await self._guard.check_output(text, self._tool_data())
            if verdict.blocked:
                self._held, self._text = [], []
                return GateStep(blocked=verdict)
        self.released_unsaved += text
        step = GateStep(release=self._held)
        self._held, self._text = [], []
        return step

    async def feed(self, event: SSEEvent) -> GateStep:
        name = event.event
        if name == "assistant_token":
            self._held.append(event)
            self._text.append(str(event.data.get("delta", "")))
            if len(self._held_text()) >= EARLY_CHECK_CHARS:
                return await self._check_held()
            return GateStep()
        if name == COMMIT_POINT:
            step = await self._check_held()
            # False when the loop keeps this text for the final row.
            if step.blocked is None and event.data.get("text_saved", True):
                self.released_unsaved = ""
            return step
        if name in ("turn_end", "error"):
            step = await self._check_held()
            if step.blocked is None:
                step.release.append(event)
            return step
        if self._held:
            self._held.append(event)
            return GateStep()
        return GateStep(release=[event])
