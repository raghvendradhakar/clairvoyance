"""Guardrails in the chat turn. With no guard (``ChatAgent._guard`` is None)
everything passes through unchanged."""

from contextlib import aclosing
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Optional,
    cast,
)

from app.ai.voice.agents.breeze_buddy.chat.history.block_codec import (
    display_text_block,
    internal_text_block,
    plain_text_blocks,
)
from app.ai.voice.agents.breeze_buddy.chat.sse import SSEEvent
from app.ai.voice.agents.breeze_buddy.guardrails.guard import COMMIT_POINT
from app.ai.voice.agents.breeze_buddy.guardrails.prompts import WITHHELD_INPUT_NOTE
from app.database.accessor.breeze_buddy.chat_session import (
    insert_chat_message,
    update_chat_session_after_turn,
)
from app.schemas.breeze_buddy.chat import ChatMessageRole

if TYPE_CHECKING:
    from app.ai.voice.agents.breeze_buddy.chat.agent.core import ChatAgent


class GuardedTurnMixin:
    async def _block_input_turn(
        self: "ChatAgent", user_content: str, reply: str
    ) -> AsyncIterator[SSEEvent]:
        """End the turn for a blocked message: the widget keeps the text, the
        LLM sees only a note. Not billed."""
        user_row = await insert_chat_message(
            session_id=self.session_id,
            role=ChatMessageRole.USER,
            content=user_content,
            content_blocks=[
                display_text_block(user_content),
                internal_text_block(WITHHELD_INPUT_NOTE),
            ],
        )
        yield SSEEvent(
            event="user_committed",
            data={"idx": user_row.idx if user_row else None, "content": user_content},
        )
        async for event in self._end_turn_with_reply(reply, node_name=None):
            yield event

    async def _guarded(
        self: "ChatAgent",
        events: AsyncIterator[SSEEvent],
        node_name: Optional[str],
        tool_data: Optional[Callable[[], Optional[str]]] = None,
    ) -> AsyncIterator[SSEEvent]:
        """Run the output gate over ``_cycle_loop``. On a block, close the loop
        before it saves or runs tools, and end with the fixed reply."""
        guard = self._guard
        if guard is None or not guard.output_enabled:
            async for event in events:
                yield event
            return
        gate = guard.output_gate(tool_data)
        blocked = False
        # Close the generator now on a block, not at garbage collection.
        cycle = cast(AsyncGenerator[SSEEvent, None], events)
        async with aclosing(cycle) as stream:
            async for event in stream:
                if event.event == "node_transition":
                    node_name = event.data.get("to") or node_name
                step = await gate.feed(event)
                for out in step.release:
                    if out.event != COMMIT_POINT:
                        yield out
                if step.blocked is not None:
                    blocked = True
                    break
        if blocked:
            # Save text already shown with the fixed reply.
            async for event in self._end_turn_with_reply(
                guard.config.output.blocked_reply,
                node_name=node_name,
                shown_before=gate.released_unsaved,
            ):
                yield event

    async def _end_turn_with_reply(
        self: "ChatAgent",
        reply: str,
        *,
        node_name: Optional[str],
        shown_before: str = "",
    ) -> AsyncIterator[SSEEvent]:
        """Show and save the fixed reply (with UI ops already shown), then end
        the turn as ACTIVE."""
        reply = reply.strip()
        shown_before = shown_before.strip()
        yield SSEEvent(
            event="assistant_token",
            data={"delta": f" {reply}" if shown_before else reply},
        )
        text = f"{shown_before} {reply}".strip()
        internal = self._internal_turn
        stored = await insert_chat_message(
            session_id=self.session_id,
            role=ChatMessageRole.ASSISTANT,
            content=None if internal else text,
            content_blocks=(
                [internal_text_block(text)] if internal else plain_text_blocks(text)
            ),
            ui_blocks=self._row_ui_blocks([]),
        )
        idx = stored.idx if stored else None
        if not internal:
            yield SSEEvent(
                event="assistant_message", data={"idx": idx, "content": text}
            )
        await update_chat_session_after_turn(
            session_id=self.session_id, current_node=node_name or None
        )
        yield SSEEvent(
            event="turn_end",
            data={"session_status": "ACTIVE", "assistant_idx": idx},
        )
