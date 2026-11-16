"""The GUARDRAIL config: ``configuration`` of the agent's evaluation_config row."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ai.voice.llm.types import LLMConfiguration, LLMProvider

RULES_MAX_CHARS = 8000
BLOCKED_REPLY_MAX_CHARS = 500


class GuardRule(BaseModel):
    """One direction: the shopper's message (input) or the assistant's
    reply (output)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # The merchant's rules, free text or Markdown.
    rules: str = Field("", max_length=RULES_MAX_CHARS)
    # Fixed text shown instead of blocked text.
    blocked_reply: str = Field("", max_length=BLOCKED_REPLY_MAX_CHARS)

    @model_validator(mode="after")
    def _reply_required_when_enabled(self) -> "GuardRule":
        if self.enabled and not self.blocked_reply.strip():
            raise ValueError("blocked_reply is required when the check is enabled")
        return self


class GuardrailsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution: Literal["sequential"] = "sequential"
    # Adds a fixed reminder before the user message. Never blocks.
    focus: bool = False
    # Judge model override; None = the template's model, as for observers.
    llm: Optional[LLMConfiguration] = None
    input: GuardRule = Field(default_factory=GuardRule)
    output: GuardRule = Field(default_factory=GuardRule)

    @model_validator(mode="after")
    def _judge_can_force_a_tool(self) -> "GuardrailsConfig":
        # call_llm(require_tool=True) is not implemented for Bedrock
        if self.llm is not None and self.llm.provider == LLMProvider.AWS_BEDROCK:
            raise ValueError("llm: Bedrock is not supported for the guardrail judge")
        return self

    @property
    def active(self) -> bool:
        return self.input.enabled or self.output.enabled or self.focus
