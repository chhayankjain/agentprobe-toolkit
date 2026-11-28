"""Steps for a scripted model: what the model returns on each call.

Build a script from these and pass it to :meth:`agentprobe.testing.Probe.model`::

    from agentprobe.testing import probe, script

    with probe() as p:
        model = p.model(
            script.tool_call("search", {"query": "x"}),
            script.reply("Done."),
            max_calls=5,
        )

Steps are served in order, one per model call. A script tests your agent's
control flow (*if* the model responds this way, does the agent cope?). It says
nothing about whether a real model would respond this way.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from agentprobe._redaction import Redactor
from agentprobe.errors import ProbeError

__all__ = ["Step", "ToolCall", "tool_call", "tool_calls", "reply", "repeat"]


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model returns. Exactly one of ``args`` / ``raw_args`` is set."""

    name: str
    args: dict[str, Any] | None
    raw_args: str | None
    id: str | None

    def describe(self, *, redactor: Redactor | None = None) -> str:
        shown = self.raw_args if self.raw_args is not None else json.dumps(self.args, default=str)
        if redactor is not None:
            shown = redactor.text(shown)
        if len(shown) > 60:
            shown = shown[:57] + "..."
        return f"{self.name}({shown})"


@dataclass(frozen=True)
class Step:
    """One scripted model response, served ``times`` times (``None`` = every later call).

    Build steps with :func:`tool_call`, :func:`tool_calls`, :func:`reply` and
    :func:`repeat`; don't construct this class directly.
    """

    kind: str  # "tool_calls" | "reply"
    calls: tuple[ToolCall, ...] = ()
    text: str = ""
    times: int | None = 1
    _validated: bool = field(default=False, repr=False, compare=False)

    def describe(self, *, redactor: Redactor | None = None) -> str:
        if self.kind == "reply":
            text = redactor.text(self.text) if redactor is not None else self.text
            text = text if len(text) <= 40 else text[:37] + "..."
            what = f"reply {text!r}"
        else:
            what = "tool call " + ", ".join(c.describe(redactor=redactor) for c in self.calls)
        if self.times is None:
            return f"repeat {what} on every call"
        if self.times != 1:
            return f"repeat {what} x{self.times}"
        return what


def _check_call(name: Any, args: Any, raw_args: Any, id: Any) -> ToolCall:
    if not isinstance(name, str) or not name:
        raise ProbeError(f"tool_call needs a tool name (a non-empty string), got {name!r}")
    if args is not None and raw_args is not None:
        raise ProbeError("tool_call takes args or raw_args, not both")
    if raw_args is not None and not isinstance(raw_args, str):
        raise ProbeError(
            f"raw_args must be a string delivered unchanged (e.g. malformed JSON), got "
            f"{type(raw_args).__name__}"
        )
    if args is not None and not isinstance(args, dict):
        raise ProbeError(f"tool_call args must be a dict, got {type(args).__name__}")
    if args is not None:
        try:
            json.dumps(args)
        except (TypeError, ValueError) as exc:
            raise ProbeError(
                f"tool_call args must be JSON-serializable (a model can only send JSON): {exc}"
            ) from exc
    if id is not None and (not isinstance(id, str) or not id):
        raise ProbeError(f"tool_call id must be a non-empty string, got {id!r}")
    frozen_args = None if raw_args is not None else copy.deepcopy(args if args is not None else {})
    return ToolCall(name=name, args=frozen_args, raw_args=raw_args, id=id)


def tool_call(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    raw_args: str | None = None,
    id: str | None = None,
) -> Step:
    """The model calls tool *name* with *args* (a JSON-serializable dict).

    ``raw_args`` instead sends the arguments as a string, delivered to your agent
    unchanged: use it for malformed JSON. ``id`` fixes the tool-call ID; by default
    each delivery gets a fresh, unique one.
    """
    return Step(kind="tool_calls", calls=(_check_call(name, args, raw_args, id),))


def tool_calls(*calls: Step) -> Step:
    """Several tool calls in one model response. Each argument is a :func:`tool_call`."""
    if not calls:
        raise ProbeError("tool_calls needs at least one tool_call(...)")
    collected: list[ToolCall] = []
    for c in calls:
        if not isinstance(c, Step) or c.kind != "tool_calls" or len(c.calls) != 1 or c.times != 1:
            raise ProbeError("tool_calls takes tool_call(...) steps only")
        collected.append(c.calls[0])
    ids = [c.id for c in collected if c.id is not None]
    if len(ids) != len(set(ids)):
        raise ProbeError("tool calls in one response need distinct ids")
    return Step(kind="tool_calls", calls=tuple(collected))


def reply(text: str) -> Step:
    """The model answers with *text* and calls no tool. ``reply("")`` is an empty reply."""
    if not isinstance(text, str):
        raise ProbeError(f"reply takes a string, got {type(text).__name__}")
    return Step(kind="reply", text=text)


def repeat(step: Step, times: int | None = None) -> Step:
    """Serve *step* ``times`` times, or on every later call when ``times`` is None.

    An unbounded repeat must be the last step and needs ``max_calls`` on the model,
    so a runaway loop fails the test instead of running forever.
    """
    if not isinstance(step, Step):
        raise ProbeError("repeat takes a step: tool_call(...), tool_calls(...) or reply(...)")
    if step.times != 1:
        raise ProbeError("repeat cannot wrap another repeat")
    if times is not None and (isinstance(times, bool) or not isinstance(times, int) or times < 1):
        raise ProbeError(f"repeat times must be a positive integer or None, got {times!r}")
    if times is not None and any(c.id is not None for c in step.calls) and times > 1:
        raise ProbeError(
            "a repeated tool_call can't have a fixed id: each delivery needs a unique id"
        )
    if times is None and any(c.id is not None for c in step.calls):
        raise ProbeError(
            "a repeated tool_call can't have a fixed id: each delivery needs a unique id"
        )
    return Step(kind=step.kind, calls=step.calls, text=step.text, times=times)
