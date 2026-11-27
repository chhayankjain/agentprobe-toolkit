"""A scripted model: deterministic model responses for testing an agent's control flow.

Create one with :meth:`agentprobe.testing.Probe.model` and hand it to your agent
through a front end:

* :meth:`ScriptedModel.langchain` - a LangChain chat model (needs ``[langchain]``).
* :meth:`ScriptedModel.serve` - a local OpenAI Chat Completions endpoint
  (``POST {base_url}/chat/completions``, non-streaming), standard library only.

Contract (see the docs page "Script the model"):

* Every request is recorded as *attempted*, then either *served* (it gets the
  next scripted response) or *rejected*. The checks and the choice of step
  happen under one lock, so at most ``max_calls`` requests are ever served,
  retries included.
* Delivering a response consumes its step. After the last step any further
  request is rejected with :class:`ScriptExhausted`.
* Scripts are sequential: a request that arrives while another is being served
  is rejected with :class:`ScriptOverlap`, never given a step at random.
* Every rejection fails the probe's gate, even if the agent catches the
  exception. Requests after the probe closed are recorded as late and are never
  evidence.

A script tests control flow (*if* the model responds this way, does the agent
cope?), not model intelligence.
"""

from __future__ import annotations

import copy
import dataclasses
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentprobe.errors import ProbeError
from agentprobe.script import Step, ToolCall

if TYPE_CHECKING:
    from agentprobe._scripted_server import ScriptedServer

#: Prefix of generated tool-call ids. Explicit ids may not use it, so a generated
#: id can never collide with one from the script.
_GENERATED_PREFIX = "agentprobe_call_"

__all__ = [
    "ModelBudgetExceeded",
    "ModelCallRecord",
    "ModelCallRejected",
    "ScriptExhausted",
    "ScriptOverlap",
    "ScriptedModel",
    "UnsupportedModelRequest",
    "ModelCallAfterProbe",
]


# ---------------------------------------------------------------------------
# Exceptions: harness failures, not simulated provider errors
# ---------------------------------------------------------------------------


class ModelCallRejected(Exception):
    """A request to a scripted model was rejected. Recorded, and fails the gate
    (except late requests), even if the agent catches it."""

    outcome = "rejected"


class ModelBudgetExceeded(ModelCallRejected):
    """The agent asked for more than ``max_calls`` model responses."""

    outcome = "budget_exceeded"


class ScriptExhausted(ModelCallRejected):
    """The agent asked for another response after the last scripted step."""

    outcome = "exhausted"


class ScriptOverlap(ModelCallRejected):
    """A request arrived while another was still being served. Scripts are
    sequential; concurrent model calls aren't supported."""

    outcome = "overlap"


class UnsupportedModelRequest(ModelCallRejected):
    """A front end received a request it doesn't support (e.g. streaming)."""

    outcome = "unsupported"


class ModelCallAfterProbe(ModelCallRejected):
    """A request arrived after the probe closed. Recorded as late, never evidence."""

    outcome = "late"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def message(
    role: str,
    content: str = "",
    *,
    name: str | None = None,
    tool_call_id: str | None = None,
    tool_call_ids: list[str] | None = None,
) -> dict[str, Any]:
    """A message as the model received it, normalised across front ends."""
    return {
        "role": role,
        "content": content,
        "name": name,
        "tool_call_id": tool_call_id,
        "tool_call_ids": list(tool_call_ids or []),
    }


@dataclass
class ModelCallRecord:
    """One request that reached a scripted model. Observed, never inferred.

    ``outcome`` is ``"served"`` or the reason it was rejected:
    ``"budget_exceeded"``, ``"exhausted"``, ``"overlap"``, ``"unsupported"`` or
    ``"late"`` (after the probe closed; never evidence).
    """

    model: int  # which scripted model in the probe (1-based)
    index: int  # request order for this model (1-based), rejected requests included
    front_end: str  # "langchain" | "server"
    outcome: str
    messages: list[dict[str, Any]]  # what the model received (see message())
    tools_offered: list[str]
    step: int | None = None  # 1-based position of the step served
    step_description: str | None = None
    returned_content: str | None = None
    returned_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    detail: str | None = None  # why it was rejected
    after_probe_end: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class Served:
    """The response a front end must deliver for one served request."""

    content: str
    tool_calls: tuple[tuple[str, ToolCall], ...]  # (id, call)


# ---------------------------------------------------------------------------
# The scripted model
# ---------------------------------------------------------------------------


class ScriptedModel:
    """Serves scripted steps in order; records every request. Create it with
    :meth:`agentprobe.testing.Probe.model`, not directly."""

    def __init__(
        self,
        steps: tuple[Step, ...],
        *,
        max_calls: int | None,
        allow_unused: bool,
        model_id: int,
    ) -> None:
        if not steps:
            raise ProbeError("a scripted model needs at least one step")
        seen_ids: dict[str, int] = {}
        for i, s in enumerate(steps, start=1):
            for c in s.calls if isinstance(s, Step) else ():
                if c.id is None:
                    continue
                if c.id.startswith(_GENERATED_PREFIX):
                    raise ProbeError(
                        f"step {i}: tool-call id {c.id!r} uses the prefix "
                        f"{_GENERATED_PREFIX!r}, which is reserved for generated ids"
                    )
                if c.id in seen_ids and seen_ids[c.id] != i:
                    raise ProbeError(
                        f"steps {seen_ids[c.id]} and {i} both use tool-call id {c.id!r}; "
                        "every tool call the model returns needs a unique id, or an "
                        "answer to one call could be mistaken for an answer to another"
                    )
                seen_ids[c.id] = i
            if not isinstance(s, Step):
                raise ProbeError(
                    f"step {i} is not a script step; build steps with script.tool_call, "
                    "script.tool_calls, script.reply or script.repeat"
                )
            if s.times is None and i != len(steps):
                raise ProbeError(
                    f"step {i} ({s.describe()}) repeats forever, so the steps after it "
                    "could never run; make it the last step"
                )
        if max_calls is not None and (
            isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1
        ):
            raise ProbeError(f"max_calls must be a positive integer or None, got {max_calls!r}")
        unbounded = steps[-1].times is None
        if unbounded and max_calls is None:
            raise ProbeError(
                "script.repeat(...) without times= needs max_calls, so a runaway loop "
                "fails the test instead of running forever"
            )
        total = None if unbounded else sum(s.times or 0 for s in steps)
        if total is not None and max_calls is not None and total > max_calls:
            raise ProbeError(
                f"the script has {total} responses but max_calls={max_calls}, so it could "
                "never be used up; raise max_calls or shorten the script"
            )
        self._steps = steps
        self._max_calls = max_calls
        self._allow_unused = allow_unused
        self._id = model_id
        self._lock = threading.Lock()
        self._delivered = [0] * len(steps)
        self._cursor = 0
        self._served = 0
        self._records: list[ModelCallRecord] = []
        self._in_flight = False
        self._closed = False
        self._servers: list[ScriptedServer] = []

    # ---- front ends ----------------------------------------------------------------

    def langchain(self) -> Any:
        """A LangChain chat model that follows this script (needs ``[langchain]``).

        Supports ``invoke``/``ainvoke`` and ``bind_tools``; not streaming."""
        try:
            from agentprobe._scripted_langchain import ScriptedChatModel
        except ModuleNotFoundError as exc:
            from agentprobe._extras import missing_extra

            raise missing_extra(exc, "ScriptedModel.langchain()", "langchain") from exc

        return ScriptedChatModel(script=self)

    def serve(self) -> ScriptedServer:
        """Start a local OpenAI Chat Completions endpoint on 127.0.0.1 that follows
        this script. Use it as a context manager; it also stops when the probe closes.

        The one supported request is ``POST {server.base_url}/chat/completions``
        without streaming. Anything else gets an HTTP 400 and fails the gate."""
        from agentprobe._scripted_server import ScriptedServer

        with self._lock:
            if self._closed:
                raise ProbeError("cannot start a server for a closed probe")
            server = ScriptedServer(self)
            self._servers.append(server)
        return server

    # ---- the request protocol (used by the front ends) -----------------------------

    def _begin(
        self, messages: list[dict[str, Any]], tools_offered: list[str], front_end: str
    ) -> tuple[ModelCallRecord, Served]:
        """Record a request and choose its response atomically. Raises a
        :class:`ModelCallRejected` subclass (after recording it) on rejection.
        A served request must be followed by :meth:`_end`."""
        with self._lock:
            index = len(self._records) + 1
            rejection: ModelCallRejected | None = None
            if self._closed:
                rejection = ModelCallAfterProbe(
                    "the probe has closed; this request was not served and is not evidence"
                )
            elif self._in_flight:
                rejection = ScriptOverlap(
                    "another request to this scripted model was still being served; "
                    "scripts are sequential and don't support concurrent model calls"
                )
            elif self._cursor >= len(self._steps):
                rejection = ScriptExhausted(
                    f"the script's {self._served} response(s) were all used; the agent "
                    f"asked for another (request {index})"
                )
            elif self._max_calls is not None and self._served >= self._max_calls:
                rejection = ModelBudgetExceeded(
                    f"max_calls={self._max_calls} reached; the agent asked for another "
                    f"response (request {index})"
                )
            if rejection is not None:
                self._records.append(
                    ModelCallRecord(
                        model=self._id,
                        index=index,
                        front_end=front_end,
                        outcome=rejection.outcome,
                        messages=messages,
                        tools_offered=tools_offered,
                        detail=str(rejection),
                        after_probe_end=rejection.outcome == "late",
                    )
                )
                raise rejection
            pos = self._cursor
            step = self._steps[pos]
            self._delivered[pos] += 1
            if step.times is not None and self._delivered[pos] >= step.times:
                self._cursor += 1
            self._served += 1
            calls = tuple(
                (c.id or f"{_GENERATED_PREFIX}{self._id}_{index}_{k}", c)
                for k, c in enumerate(step.calls, start=1)
            )
            served = Served(content=step.text if step.kind == "reply" else "", tool_calls=calls)
            record = ModelCallRecord(
                model=self._id,
                index=index,
                front_end=front_end,
                outcome="served",
                messages=messages,
                tools_offered=tools_offered,
                step=pos + 1,
                step_description=step.describe(),
                returned_content=served.content,
                returned_tool_calls=[
                    {
                        "id": cid,
                        "name": c.name,
                        "args": copy.deepcopy(c.args),
                        "raw_args": c.raw_args,
                    }
                    for cid, c in calls
                ],
            )
            self._records.append(record)
            self._in_flight = True
            return record, served

    def _end(self) -> None:
        with self._lock:
            self._in_flight = False

    def _reject_unsupported(
        self,
        messages: list[dict[str, Any]],
        tools_offered: list[str],
        front_end: str,
        detail: str,
    ) -> ModelCallRejected:
        """Record an unsupported request and return the exception describing it."""
        with self._lock:
            late = self._closed
            exc: ModelCallRejected = (
                ModelCallAfterProbe(f"after the probe closed: {detail}")
                if late
                else UnsupportedModelRequest(detail)
            )
            self._records.append(
                ModelCallRecord(
                    model=self._id,
                    index=len(self._records) + 1,
                    front_end=front_end,
                    outcome=exc.outcome,
                    messages=messages,
                    tools_offered=tools_offered,
                    detail=str(exc),
                    after_probe_end=late,
                )
            )
        return exc

    # ---- evidence ----------------------------------------------------------------------

    @property
    def calls(self) -> list[ModelCallRecord]:
        """Snapshot of every request, in arrival order."""
        with self._lock:
            # Deep copies: editing a report must never rewrite the observations.
            return [copy.deepcopy(r) for r in self._records]

    def problems(self) -> list[str]:
        """Gate failures: rejected requests (except late ones) and unused steps."""
        with self._lock:
            out: list[str] = []
            for r in self._records:
                if r.outcome not in ("served", "late"):
                    out.append(f"scripted model {self._id}, request {r.index}: {r.detail}")
            if not self._allow_unused:
                ended = f"the run ended after {self._served} served model call(s)"
                for i, (step, n) in enumerate(zip(self._steps, self._delivered), start=1):
                    if step.times is None and n == 0:
                        out.append(
                            f"scripted model {self._id}, step {i} ({step.describe()}) "
                            f"unused: {ended}"
                        )
                    elif step.times is not None and n < step.times:
                        used = "unused" if n == 0 else f"delivered {n} of {step.times}"
                        out.append(
                            f"scripted model {self._id}, step {i} ({step.describe()}) "
                            f"{used}: {ended}"
                        )
            return out

    def _close(self) -> None:
        with self._lock:
            self._closed = True
            servers = list(self._servers)
        for s in servers:
            s.close()
