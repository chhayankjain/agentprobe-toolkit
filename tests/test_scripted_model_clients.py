"""Scripted model server with real non-LangChain clients (issue #131 acceptance).

* a plain OpenAI SDK tool loop (openai>=1)
* pydantic-ai's OpenAI Chat Completions model

Both are installed in CI; locally the tests skip if the client is missing.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from agentprobe.testing import ModelBudgetExceeded, probe, script

#: CI sets this so the acceptance tests can never silently skip there.
REQUIRED = os.environ.get("AGENTPROBE_REQUIRE_CLIENTS") == "1"

if REQUIRED:
    import openai
else:
    openai = pytest.importorskip("openai")
if int(openai.__version__.split(".")[0]) < 1:
    if REQUIRED:
        raise RuntimeError(f"needs openai>=1, found {openai.__version__}")
    pytest.skip("needs openai>=1", allow_module_level=True)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "order_status",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def sdk_agent(client: Any, question: str, *, report_errors: bool) -> str:
    """A plain OpenAI SDK tool loop. With report_errors=False it lets a bad
    tool call crash the run (the bug class the scripted model should expose)."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    for _ in range(10):
        resp = client.chat.completions.create(model="any", messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return str(msg.content)
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments)
                out = f"order {args['order_id']}: shipped"
            except (ValueError, KeyError) as e:
                if not report_errors:
                    raise
                out = f"error: {type(e).__name__}: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
    return "gave up"


def _client(server: Any) -> Any:
    return openai.OpenAI(base_url=server.base_url, api_key=server.api_key)


@pytest.mark.parametrize("report_errors", [True, False], ids=["handles", "broken-control"])
def test_openai_sdk_malformed_arguments(report_errors):
    with probe(enforce=False) as p:
        m = p.model(
            script.tool_call("order_status", raw_args='{"order_id": '),
            script.reply("Sorry, I could not check that order."),
            max_calls=3,
        )
        with m.serve() as server:
            client = _client(server)
            r = p.run(lambda q: sdk_agent(client, q, report_errors=report_errors), "Order 42?")
        r.assert_no_crash(required=False)
        r.assert_model_received("JSONDecodeError", role="tool", required=False)
        r.assert_every_tool_call_answered(required=False)
    statuses = [a.status for a in r.assertions]
    if report_errors:
        assert statuses == ["pass", "pass", "pass"]
        assert p.gate().passed
    else:
        assert statuses == ["fail", "unassessed", "unassessed"]
        assert any("step 2" in f for f in p.gate().failures)


def test_openai_sdk_budget_is_not_retried_around():
    with probe(enforce=False) as p:
        m = p.model(script.repeat(script.tool_call("order_status", {"order_id": "1"})), max_calls=3)
        with m.serve() as server:
            client = _client(server)  # the SDK retries 429/5xx by default, not 400
            r = p.run(lambda q: sdk_agent(client, q, report_errors=True), "loop")
    assert isinstance(r.exception, openai.BadRequestError)
    assert "max_calls=3 reached" in str(r.exception)
    assert [c.outcome for c in m.calls] == ["served"] * 3 + ["budget_exceeded"]
    assert not p.gate().passed
    assert ModelBudgetExceeded.outcome == "budget_exceeded"


def test_openai_sdk_streaming_is_rejected():
    with probe(enforce=False) as p:
        m = p.model(script.reply("x"), allow_unused=True)
        with m.serve() as server, pytest.raises(openai.BadRequestError, match="streaming"):
            _client(server).chat.completions.create(
                model="any", messages=[{"role": "user", "content": "x"}], stream=True
            )
    assert any("streaming is not supported" in f for f in p.gate().failures)


def test_pydantic_ai_agent_recovers_from_bad_arguments(monkeypatch):
    if REQUIRED:
        import pydantic_ai
    else:
        pydantic_ai = pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    with probe() as p:
        m = p.model(
            script.tool_call("order_status", {"order": "42"}),  # wrong argument name
            script.tool_call("order_status", {"order_id": "42"}),
            script.reply("Order 42 has shipped."),
            max_calls=5,
        )
        with m.serve() as server:
            model = OpenAIChatModel(
                "scripted",
                provider=OpenAIProvider(base_url=server.base_url, api_key=server.api_key),
            )
            agent = pydantic_ai.Agent(model)

            @agent.tool_plain
            def order_status(order_id: str) -> str:
                return f"order {order_id}: shipped"

            r = p.run(lambda q: agent.run_sync(q).output, "Where is order 42?")
        r.assert_no_crash()
        r.assert_model_received("order_id", role="tool")
        r.assert_every_tool_call_answered()
        r.assert_answer_contains("shipped")
