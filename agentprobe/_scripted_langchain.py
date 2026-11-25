"""LangChain front end for :class:`agentprobe.scripted_model.ScriptedModel`.

Get it with ``model.langchain()``. Needs ``langchain-core`` (the ``[langchain]``
extra). Not streaming.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from agentprobe._text import message_text
from agentprobe.scripted_model import ScriptedModel, Served, message


def _normalize(m: BaseMessage) -> dict[str, Any]:
    text = message_text(m)
    if isinstance(m, ToolMessage):
        return message("tool", text, name=m.name, tool_call_id=m.tool_call_id)
    if isinstance(m, AIMessage):
        ids = [c.get("id") or "" for c in m.tool_calls]
        ids += [c.get("id") or "" for c in m.invalid_tool_calls]
        return message("assistant", text, tool_call_ids=[i for i in ids if i])
    if isinstance(m, SystemMessage):
        return message("system", text)
    if isinstance(m, HumanMessage):
        return message("user", text)
    return message(getattr(m, "type", "unknown"), text)


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function")
            name = fn.get("name") if isinstance(fn, dict) else t.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def _ai_message(served: Served) -> AIMessage:
    tool_calls: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for cid, call in served.tool_calls:
        arguments = call.raw_args if call.raw_args is not None else json.dumps(call.args)
        raw.append(
            {"id": cid, "type": "function", "function": {"name": call.name, "arguments": arguments}}
        )
        # Mirror how LangChain's provider integrations report arguments: parsed when
        # they are a JSON object, otherwise an invalid tool call with the raw string.
        try:
            parsed = json.loads(arguments)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            tool_calls.append({"name": call.name, "args": parsed, "id": cid, "type": "tool_call"})
        else:
            invalid.append(
                {
                    "name": call.name,
                    "args": arguments,
                    "id": cid,
                    "error": "arguments are not a JSON object",
                    "type": "invalid_tool_call",
                }
            )
    return AIMessage(
        content=served.content,
        tool_calls=tool_calls,
        invalid_tool_calls=invalid,
        additional_kwargs={"tool_calls": raw} if raw else {},
        response_metadata={
            "finish_reason": "tool_calls" if raw else "stop",
            "model_name": "agentprobe-scripted",
        },
    )


class ScriptedChatModel(BaseChatModel):
    """A LangChain chat model that follows a script. Every request is recorded on
    the script; rejected requests raise a ``ModelCallRejected`` subclass."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: ScriptedModel = Field(exclude=True)

    @property
    def _llm_type(self) -> str:
        return "agentprobe-scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)

    def _serve(self, messages: list[BaseMessage], kwargs: dict[str, Any]) -> ChatResult:
        normalized = [_normalize(m) for m in messages]
        _, served = self.script._begin(normalized, _tool_names(kwargs.get("tools")), "langchain")
        try:
            return ChatResult(generations=[ChatGeneration(message=_ai_message(served))])
        finally:
            self.script._end()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._serve(messages, kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._serve(messages, kwargs)


__all__ = ["ScriptedChatModel"]
