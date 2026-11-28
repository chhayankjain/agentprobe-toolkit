"""pytest-friendly entry point for fault tests on your own agent.

Use :func:`probe` as a context manager in any test::

    from agentprobe.testing import probe

    def test_order_status_survives_a_timeout():
        with probe() as p:
            tools = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1, with_="timeout")
            result = p.run(lambda q: answer(q, tools), "Did order 42 go through?")
            result.assert_answer_contains("confirmed")

Or as a fixture in ``conftest.py``::

    import pytest
    from agentprobe.testing import probe as _probe

    @pytest.fixture
    def probe():
        with _probe() as p:
            yield p

To test what your agent does with model responses (bad tool arguments, unknown
tools, loops, empty replies), script the model with ``p.model(...)`` and the
steps in :mod:`agentprobe.script`.

When the ``with`` block ends normally, the gate runs: the test fails unless every
required scenario was exercised and every required assertion passed. A full pytest
plugin with markers and reports is planned for v1.5.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentprobe import script
from agentprobe._redaction import RedactionPattern
from agentprobe.faults import (
    ToolCallAPIError,
    ToolCallError,
    ToolCallMalformedOutput,
    ToolCallRateLimited,
    ToolCallTimeout,
)
from agentprobe.probe import (
    SCHEMA_VERSION,
    AssertionRecord,
    CallRecord,
    GateReport,
    Probe,
    ProbeError,
    ProbeGateError,
    ProbeResult,
    ScenarioEvidence,
    ToolScenarios,
)
from agentprobe.scripted_model import (
    ModelBudgetExceeded,
    ModelCallAfterProbe,
    ModelCallRecord,
    ModelCallRejected,
    ScriptedModel,
    ScriptExhausted,
    ScriptOverlap,
    UnsupportedModelRequest,
)


def probe(
    *,
    enforce: bool = True,
    redact: bool = True,
    redact_keys: Sequence[str] = (),
    redact_patterns: Sequence[RedactionPattern] = (),
) -> Probe:
    """Create a probe. ``enforce=False`` skips the gate on exit (for exploration);
    you can still call ``result.gate()`` yourself. Reports redact by default;
    ``redact=False`` opts out, ``redact_patterns`` adds regex patterns, and ``redact_keys``
    adds sensitive field names. Execution and comparison evidence are unchanged."""
    return Probe(
        enforce=enforce, redact=redact, redact_keys=redact_keys, redact_patterns=redact_patterns
    )


__all__ = [
    "probe",
    "Probe",
    "ProbeResult",
    "ProbeError",
    "ProbeGateError",
    "ToolScenarios",
    "CallRecord",
    "ScenarioEvidence",
    "AssertionRecord",
    "GateReport",
    "SCHEMA_VERSION",
    # Scripted models: p.model(script.tool_call(...), script.reply(...))
    "script",
    "ScriptedModel",
    "ModelCallRecord",
    "ModelCallRejected",
    "ModelBudgetExceeded",
    "ScriptExhausted",
    "ScriptOverlap",
    "UnsupportedModelRequest",
    "ModelCallAfterProbe",
    # The built-in faults, for agents and tests that catch them
    "ToolCallError",
    "ToolCallTimeout",
    "ToolCallRateLimited",
    "ToolCallAPIError",
    "ToolCallMalformedOutput",
]
