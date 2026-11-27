"""Scripted models (issue #131): steps, budget, consumption, gate, assertions, front ends.

Every gate rule and assertion has a deliberately broken control that must fail
for the intended reason, next to the case that passes.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
import threading
import urllib.error
import urllib.request
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentprobe import _scripted_langchain
from agentprobe.testing import (
    ModelBudgetExceeded,
    ModelCallAfterProbe,
    ProbeError,
    ProbeGateError,
    ScriptExhausted,
    ScriptOverlap,
    UnsupportedModelRequest,
    probe,
    script,
)

# ---------------------------------------------------------------------------
# A tiny tool-calling agent over a LangChain chat model
# ---------------------------------------------------------------------------


def lookup(order_id: str) -> str:
    return f"order {order_id}: shipped"


def agent_loop(llm: Any, question: str, *, max_turns: int = 10, answer_tools: bool = True) -> str:
    """Calls the model; runs tool calls; feeds results back. Tool errors (bad args,
    unknown tool) go back to the model as tool messages when ``answer_tools``."""
    bound = llm.bind_tools([lookup])
    messages: list[Any] = [HumanMessage(question)]
    for _ in range(max_turns):
        ai = bound.invoke(messages)
        messages.append(ai)
        calls = list(ai.tool_calls) + list(ai.invalid_tool_calls)
        if not calls:
            return str(ai.content)
        if not answer_tools:
            raise RuntimeError("this agent can't run tools")
        for c in calls:
            try:
                if c["name"] != "lookup" or not isinstance(c["args"], dict):
                    raise ValueError(f"bad tool call: {c['name']}({c['args']!r})")
                out = lookup(**c["args"])
            except Exception as e:  # noqa: BLE001
                out = f"error: {e}"
            messages.append(ToolMessage(out, tool_call_id=c["id"]))
    return "gave up"


def gate_failures(p: Any) -> list[str]:
    return p.gate().failures


# ---------------------------------------------------------------------------
# Declaring scripts
# ---------------------------------------------------------------------------


class TestDeclaration:
    def test_needs_a_step(self):
        with probe(enforce=False) as p, pytest.raises(ProbeError, match="at least one step"):
            p.model()

    def test_rejects_non_steps(self):
        with probe(enforce=False) as p, pytest.raises(ProbeError, match="not a script step"):
            p.model("reply")  # type: ignore[arg-type]

    def test_unbounded_repeat_must_be_last(self):
        with probe(enforce=False) as p, pytest.raises(ProbeError, match="make it the last"):
            p.model(script.repeat(script.reply("x")), script.reply("y"), max_calls=5)

    def test_unbounded_repeat_needs_max_calls(self):
        with probe(enforce=False) as p, pytest.raises(ProbeError, match="needs max_calls"):
            p.model(script.repeat(script.reply("x")))

    def test_script_longer_than_budget_is_rejected(self):
        with probe(enforce=False) as p, pytest.raises(ProbeError, match="never be used up"):
            p.model(script.reply("a"), script.repeat(script.reply("b"), times=3), max_calls=3)

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: script.tool_call(""), "tool name"),
            (lambda: script.tool_call("t", {"a": 1}, raw_args="{}"), "not both"),
            (lambda: script.tool_call("t", raw_args=b"{}"), "raw_args must be a string"),  # type: ignore[arg-type]
            (lambda: script.tool_call("t", ["a"]), "must be a dict"),  # type: ignore[arg-type]
            (lambda: script.tool_call("t", {"a": object()}), "JSON-serializable"),
            (lambda: script.tool_calls(), "at least one"),
            (lambda: script.tool_calls(script.reply("x")), "tool_call"),
            (
                lambda: script.tool_calls(
                    script.tool_call("a", id="same"), script.tool_call("b", id="same")
                ),
                "distinct ids",
            ),
            (lambda: script.repeat(script.tool_call("t", id="fixed"), times=2), "fixed id"),
            (lambda: script.repeat(script.repeat(script.reply("x"), times=2)), "another repeat"),
            (lambda: script.repeat(script.reply("x"), times=0), "positive integer"),
            (lambda: script.reply(None), "takes a string"),  # type: ignore[arg-type]
        ],
    )
    def test_bad_steps(self, make, match):
        with pytest.raises(ProbeError, match=match):
            make()

    def test_bad_max_calls(self):
        with probe(enforce=False) as p, pytest.raises(ProbeError, match="max_calls"):
            p.model(script.reply("x"), max_calls=0)

    def test_args_are_copied_at_declaration(self):
        args = {"order_id": "1"}
        step = script.tool_call("lookup", args)
        args["order_id"] = "changed"
        assert step.calls[0].args == {"order_id": "1"}

    def test_no_model_after_close(self):
        p = probe()
        with p:
            pass
        with pytest.raises(ProbeError, match="closed probe"):
            p.model(script.reply("x"))


# ---------------------------------------------------------------------------
# Consumption and the gate
# ---------------------------------------------------------------------------


class TestConsumption:
    def test_full_script_passes(self):
        with probe() as p:
            m = p.model(
                script.tool_call("lookup", {"order_id": "42"}),
                script.reply("Shipped."),
                max_calls=5,
            )
            r = p.run(lambda q: agent_loop(m.langchain(), q), "Where is 42?")
            r.assert_answer_contains("shipped")
        assert [c.outcome for c in m.calls] == ["served", "served"]

    def test_unused_step_fails_the_gate(self):
        with pytest.raises(ProbeGateError, match=r"step 2 \(reply 'never'\) unused"):
            with probe() as p:
                m = p.model(script.reply("only"), script.reply("never"))
                p.run(lambda q: agent_loop(m.langchain(), q), "hi")

    def test_unused_message_says_when_the_run_ended(self):
        with probe(enforce=False) as p:
            m = p.model(script.reply("only"), script.reply("never"))
            p.run(lambda q: agent_loop(m.langchain(), q), "hi")
        assert "the run ended after 1 served model call(s)" in gate_failures(p)[0]

    def test_allow_unused(self):
        with probe() as p:
            m = p.model(script.reply("only"), script.reply("never"), allow_unused=True)
            p.run(lambda q: agent_loop(m.langchain(), q), "hi").assert_no_crash()

    def test_partially_delivered_repeat_is_reported(self):
        with probe(enforce=False) as p:
            m = p.model(script.repeat(script.reply("x"), times=3))
            llm = m.langchain()
            llm.invoke("a")
            llm.invoke("b")
        assert "delivered 2 of 3" in gate_failures(p)[0]

    def test_finite_repeat_then_next_step(self):
        with probe() as p:
            m = p.model(
                script.repeat(script.tool_call("lookup", {"order_id": "1"}), times=2),
                script.reply("done"),
            )
            r = p.run(lambda q: agent_loop(m.langchain(), q), "x")
            r.assert_answer_contains("done")
        ids = [c.returned_tool_calls[0]["id"] for c in m.calls[:2]]
        assert len(set(ids)) == 2  # each delivery gets a fresh id

    def test_extra_request_is_exhausted_and_fails_the_gate(self):
        with probe(enforce=False) as p:
            m = p.model(script.reply("one"))
            llm = m.langchain()
            llm.invoke("a")
            with pytest.raises(ScriptExhausted):
                llm.invoke("b")
        assert [c.outcome for c in m.calls] == ["served", "exhausted"]
        assert any("were all used" in f for f in gate_failures(p))

    def test_normal_end_after_last_step_is_fine(self):
        with probe() as p:
            m = p.model(script.reply("one"))
            m.langchain().invoke("a")


class TestBudget:
    def test_budget_stops_a_loop(self):
        with probe(enforce=False) as p:
            m = p.model(script.repeat(script.tool_call("lookup", {"order_id": "1"})), max_calls=3)
            r = p.run(lambda q: agent_loop(m.langchain(), q, max_turns=100), "x")
        assert isinstance(r.exception, ModelBudgetExceeded)
        assert [c.outcome for c in m.calls] == ["served"] * 3 + ["budget_exceeded"]
        assert any("max_calls=3 reached" in f for f in gate_failures(p))

    def test_budget_violation_fails_the_gate_even_if_the_agent_catches_it(self):
        def swallowing_agent(llm: Any, q: str) -> str:
            for _ in range(10):
                try:
                    llm.invoke(q)
                except Exception:  # noqa: BLE001 - the agent hides the failure
                    return "all good"
            return "all good"

        with probe(enforce=False) as p:
            m = p.model(script.repeat(script.reply("again")), max_calls=2)
            r = p.run(lambda q: swallowing_agent(m.langchain(), q), "x")
            r.assert_no_crash()  # the agent looks fine...
        assert any("max_calls=2 reached" in f for f in gate_failures(p))  # ...the gate doesn't

    def test_never_serves_more_than_max_calls_under_concurrency(self):
        with probe(enforce=False) as p:
            m = p.model(script.repeat(script.reply("x")), max_calls=5)
            llm = m.langchain()
            errors: list[BaseException] = []

            def hammer():
                for _ in range(20):
                    try:
                        llm.invoke("x")
                    except Exception as e:  # noqa: BLE001
                        errors.append(e)

            threads = [threading.Thread(target=hammer) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert sum(c.outcome == "served" for c in m.calls) == 5
        assert len(m.calls) == 80

    def test_assert_model_calls_at_most(self):
        with probe() as p:
            m = p.model(script.reply("x"))
            r = p.run(lambda q: agent_loop(m.langchain(), q), "x")
            r.assert_model_calls_at_most(1)

    def test_assert_model_calls_at_most_broken_control(self):
        with probe(enforce=False) as p:
            m = p.model(script.tool_call("lookup", {"order_id": "1"}), script.reply("x"))
            r = p.run(lambda q: agent_loop(m.langchain(), q), "x")
            with pytest.raises(AssertionError, match=r"2 request\(s\), 2 served"):
                r.assert_model_calls_at_most(1)


class TestOverlap:
    def test_concurrent_request_is_rejected_not_assigned(self, monkeypatch):
        entered, release = threading.Event(), threading.Event()
        real = _scripted_langchain._ai_message

        def slow_ai_message(served):
            entered.set()
            release.wait(5)
            return real(served)

        monkeypatch.setattr(_scripted_langchain, "_ai_message", slow_ai_message)
        with probe(enforce=False) as p:
            m = p.model(script.reply("first"), script.reply("second"))
            llm = m.langchain()
            out: list[Any] = []
            t = threading.Thread(target=lambda: out.append(llm.invoke("a")))
            t.start()
            assert entered.wait(5)
            with pytest.raises(ScriptOverlap):
                llm.invoke("b")
            release.set()
            t.join(5)
        assert out[0].content == "first"
        assert [c.outcome for c in m.calls] == ["served", "overlap"]
        failures = gate_failures(p)
        assert any("sequential" in f for f in failures)
        assert any("step 2" in f and "unused" in f for f in failures)


class TestLateAndIsolation:
    def test_request_after_close_is_late_and_not_a_gate_failure(self):
        with probe() as p:
            m = p.model(script.reply("x"))
            llm = m.langchain()
            llm.invoke("a")
        with pytest.raises(ModelCallAfterProbe):
            llm.invoke("b")
        late = m.calls[-1]
        assert late.outcome == "late" and late.after_probe_end
        assert p.gate().passed

    def test_two_probes_do_not_share_state(self):
        with probe() as p1, probe() as p2:
            a = p1.model(script.reply("from 1")).langchain()
            b = p2.model(script.reply("from 2")).langchain()
            assert b.invoke("x").content == "from 2"
            assert a.invoke("x").content == "from 1"


# ---------------------------------------------------------------------------
# Assertions: not observed (unassessed) vs confirmed absent (fail)
# ---------------------------------------------------------------------------


class TestEveryToolCallAnswered:
    def test_passes_when_results_come_back(self):
        with probe() as p:
            m = p.model(
                script.tool_calls(
                    script.tool_call("lookup", {"order_id": "1"}),
                    script.tool_call("lookup", {"order_id": "2"}),
                ),
                script.reply("both shipped"),
            )
            r = p.run(lambda q: agent_loop(m.langchain(), q), "x")
            r.assert_every_tool_call_answered()

    def test_broken_control_one_of_two_results_dropped(self):
        def drops_second(llm: Any, q: str) -> str:
            bound = llm.bind_tools([lookup])
            ai = bound.invoke([HumanMessage(q)])
            first = ai.tool_calls[0]
            return str(
                bound.invoke(
                    [HumanMessage(q), ai, ToolMessage("shipped", tool_call_id=first["id"])]
                ).content
            )

        with probe(enforce=False) as p:
            m = p.model(
                script.tool_calls(
                    script.tool_call("lookup", {"order_id": "1"}, id="a"),
                    script.tool_call("lookup", {"order_id": "2"}, id="b"),
                ),
                script.reply("done"),
            )
            r = p.run(lambda q: drops_second(m.langchain(), q), "x")
            with pytest.raises(AssertionError, match=r"never included a tool result for: b "):
                r.assert_every_tool_call_answered()
        assert r.assertions[-1].status == "fail"

    def test_unassessed_when_no_later_request(self):
        with probe(enforce=False) as p:
            m = p.model(script.tool_call("lookup", {"order_id": "1"}))
            r = p.run(lambda q: agent_loop(m.langchain(), q, answer_tools=False), "x")
            with pytest.raises(AssertionError, match="never observable"):
                r.assert_every_tool_call_answered()
        assert r.assertions[-1].status == "unassessed"

    def test_unassessed_when_no_tool_calls(self):
        with probe(enforce=False) as p:
            m = p.model(script.reply("x"))
            r = p.run(lambda q: agent_loop(m.langchain(), q), "x")
            with pytest.raises(AssertionError, match="nothing to check"):
                r.assert_every_tool_call_answered()
        assert r.assertions[-1].status == "unassessed"


class TestModelReceived:
    def _run(self, p: Any, *, bad_args: bool = True) -> Any:
        args = {"order": "1"} if bad_args else {"order_id": "1"}
        m = p.model(script.tool_call("lookup", args), script.reply("done"))
        return p.run(lambda q: agent_loop(m.langchain(), q), "x")

    def test_passes_when_the_error_reaches_the_model(self):
        with probe() as p:
            self._run(p).assert_model_received("unexpected keyword", role="tool")

    def test_broken_control_absent_from_later_requests(self):
        with probe(enforce=False) as p:
            r = self._run(p, bad_args=False)
            with pytest.raises(AssertionError, match=r"not in any of 2 model request"):
                r.assert_model_received("unexpected keyword", role="tool")
        assert r.assertions[-1].status == "fail"

    def test_role_filter(self):
        with probe(enforce=False) as p:
            r = self._run(p)
            with pytest.raises(AssertionError):
                r.assert_model_received("unexpected keyword", role="user")

    def test_unassessed_when_the_run_ended_after_tool_calls(self):
        with probe(enforce=False) as p:
            m = p.model(script.tool_call("lookup", {"order": "1"}))
            r = p.run(lambda q: agent_loop(m.langchain(), q, answer_tools=False), "x")
            with pytest.raises(AssertionError, match="never observable"):
                r.assert_model_received("unexpected keyword")
        assert r.assertions[-1].status == "unassessed"

    def test_needs_a_scripted_model(self):
        with probe(enforce=False) as p:
            r = p.run(lambda q: "x", "x")
            with pytest.raises(ProbeError, match="needs a scripted model"):
                r.assert_model_received("x")


# ---------------------------------------------------------------------------
# LangChain front end
# ---------------------------------------------------------------------------


class TestLangChainFrontEnd:
    def test_malformed_raw_args_arrive_unchanged(self):
        raw = '{"order_id": '
        with probe(enforce=False) as p:
            ai = p.model(script.tool_call("lookup", raw_args=raw)).langchain().invoke("x")
        assert ai.tool_calls == []
        assert ai.invalid_tool_calls[0]["args"] == raw
        assert ai.additional_kwargs["tool_calls"][0]["function"]["arguments"] == raw

    def test_raw_args_that_are_a_json_object_parse_like_a_provider(self):
        with probe(enforce=False) as p:
            ai = (
                p.model(script.tool_call("lookup", raw_args='{"order_id":"1"}'))
                .langchain()
                .invoke("x")
            )
        assert ai.tool_calls[0]["args"] == {"order_id": "1"}
        assert ai.additional_kwargs["tool_calls"][0]["function"]["arguments"] == '{"order_id":"1"}'

    def test_records_messages_and_offered_tools(self):
        with probe(enforce=False) as p:
            m = p.model(script.tool_call("lookup", {"order_id": "1"}), script.reply("done"))
            r = p.run(lambda q: agent_loop(m.langchain(), q), "Where?")
        first, second = r.model_calls
        assert first.tools_offered == ["lookup"]
        assert first.messages[0]["role"] == "user" and first.messages[0]["content"] == "Where?"
        tool_msg = second.messages[-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == first.returned_tool_calls[0]["id"]
        assert second.messages[1]["tool_call_ids"] == [first.returned_tool_calls[0]["id"]]

    def test_empty_reply(self):
        with probe(enforce=False) as p:
            ai = p.model(script.reply("")).langchain().invoke("x")
        assert isinstance(ai, AIMessage) and ai.content == "" and not ai.tool_calls

    async def test_async(self):
        async with probe() as p:
            m = p.model(script.reply("async ok"))
            ai = await m.langchain().ainvoke("x")
            assert ai.content == "async ok"


# ---------------------------------------------------------------------------
# Server front end (standard library HTTP; the OpenAI SDK is tested separately)
# ---------------------------------------------------------------------------


def post(base_url: str, body: Any, path: str = "/chat/completions") -> tuple[int, dict[str, Any]]:
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(
        base_url + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


USER = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


class TestServerFrontEnd:
    def test_binds_loopback_and_serves(self):
        with probe() as p:
            m = p.model(script.reply("hello"))
            with m.serve() as s:
                assert s.host == "127.0.0.1" and s.base_url.endswith("/v1")
                status, body = post(s.base_url, USER)
        assert status == 200
        assert body["choices"][0]["message"]["content"] == "hello"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_raw_args_byte_for_byte(self):
        raw = '{"order_id": "42",, }'
        with probe(enforce=False) as p:
            m = p.model(script.tool_call("lookup", raw_args=raw, id="c1"))
            with m.serve() as s:
                _, body = post(s.base_url, USER)
        call = body["choices"][0]["message"]["tool_calls"][0]
        assert call == {
            "id": "c1",
            "type": "function",
            "function": {"name": "lookup", "arguments": raw},
        }

    def test_tool_round_trip_is_recorded(self):
        with probe() as p:
            m = p.model(script.tool_call("lookup", {"order_id": "1"}, id="c1"), script.reply("ok"))
            with m.serve() as s:
                post(
                    s.base_url,
                    {**USER, "tools": [{"type": "function", "function": {"name": "lookup"}}]},
                )
                post(
                    s.base_url,
                    {
                        "model": "m",
                        "messages": [
                            {"role": "user", "content": "hi"},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "lookup", "arguments": "{}"},
                                    }
                                ],
                            },
                            {"role": "tool", "tool_call_id": "c1", "content": "shipped"},
                        ],
                    },
                )
            r = p.run(lambda q: "ok", "x")
            r.assert_every_tool_call_answered()
        assert m.calls[0].tools_offered == ["lookup"]

    @pytest.mark.parametrize(
        "body, path, match",
        [
            ({**USER, "stream": True}, "/chat/completions", "streaming is not supported"),
            ({**USER, "n": 2}, "/chat/completions", "one choice only"),
            (USER, "/responses", "implements only POST /v1/chat/completions"),
            (b"{not json", "/chat/completions", "not valid JSON"),
        ],
    )
    def test_unsupported_requests_fail_clearly_and_fail_the_gate(self, body, path, match):
        with probe(enforce=False) as p:
            m = p.model(script.reply("x"), allow_unused=True)
            with m.serve() as s:
                status, resp = post(s.base_url, body, path)
        assert status == 400
        assert resp["error"]["code"] == "unsupported" and match in resp["error"]["message"]
        assert m.calls[0].outcome == "unsupported"
        assert any(match in f for f in gate_failures(p))

    def test_get_is_unsupported(self):
        with probe(enforce=False) as p:
            m = p.model(script.reply("x"), allow_unused=True)
            with m.serve() as s:
                with pytest.raises(urllib.error.HTTPError) as err:
                    urllib.request.urlopen(s.base_url + "/models", timeout=5)
        assert err.value.code == 400
        assert m.calls[0].outcome == "unsupported"

    def test_retrying_client_cannot_escape_the_budget(self):
        with probe(enforce=False) as p:
            m = p.model(script.repeat(script.reply("x")), max_calls=2)
            with m.serve() as s:
                statuses = [post(s.base_url, USER)[0] for _ in range(6)]  # a client retrying
        assert statuses == [200, 200, 400, 400, 400, 400]
        assert sum(c.outcome == "served" for c in m.calls) == 2
        assert sum(c.outcome == "budget_exceeded" for c in m.calls) == 4

    def test_server_stops_when_the_probe_closes(self):
        with probe(enforce=False) as p:
            m = p.model(script.reply("x"), allow_unused=True)
            s = m.serve()  # no with-block: the probe must clean up
            port = s.port
        assert s.closed
        with pytest.raises(OSError), socket.create_connection(("127.0.0.1", port), timeout=1):
            pass

    def test_server_stops_when_the_test_fails(self):
        port = None
        with pytest.raises(RuntimeError):
            with probe() as p:
                m = p.model(script.reply("x"))
                port = m.serve().port
                raise RuntimeError("test failed")
        with pytest.raises(OSError), socket.create_connection(("127.0.0.1", port), timeout=1):
            pass

    def test_exhausted_over_http(self):
        with probe(enforce=False) as p:
            m = p.model(script.reply("x"))
            with m.serve() as s:
                post(s.base_url, USER)
                status, resp = post(s.base_url, USER)
        assert status == 400 and resp["error"]["code"] == "exhausted"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def test_to_dict_includes_model_calls():
    with probe(enforce=False) as p:
        m = p.model(script.tool_call("lookup", raw_args="{"), script.reply("x"), max_calls=2)
        r = p.run(lambda q: agent_loop(m.langchain(), q), "x")
    d = json.loads(json.dumps(r.to_dict()))
    assert d["schema_version"] == "1.3"
    calls = d["observed"]["model_calls"]
    assert [c["outcome"] for c in calls] == ["served", "served"]
    assert calls[0]["returned_tool_calls"][0]["raw_args"] == "{"
    assert "model calls:" in r.summary()


# ---------------------------------------------------------------------------
# Review findings on #132: each was a false pass before the fix
# ---------------------------------------------------------------------------


class TestToolCallIdsAreUnique:
    def test_same_explicit_id_in_two_steps_is_rejected(self):
        """Before: answering the first call and omitting the second passed, because
        the old result looked like an answer to the new call."""
        with probe(enforce=False) as p, pytest.raises(ProbeError, match="both use tool-call id"):
            p.model(
                script.tool_call("lookup", {"order_id": "1"}, id="dup"),
                script.tool_call("lookup", {"order_id": "2"}, id="dup"),
                script.reply("x"),
            )

    def test_explicit_id_cannot_use_the_generated_prefix(self):
        with probe(enforce=False) as p, pytest.raises(ProbeError, match="reserved"):
            p.model(script.tool_call("lookup", id="agentprobe_call_1_1_1"), script.reply("x"))

    def test_generated_ids_use_the_reserved_prefix_and_are_unique(self):
        with probe(enforce=False) as p:
            m = p.model(
                script.repeat(script.tool_call("lookup", {"order_id": "1"}), times=3),
                script.reply("x"),
            )
            r = p.run(lambda q: agent_loop(m.langchain(), q), "x")
        ids = [c["id"] for rec in r.model_calls for c in rec.returned_tool_calls]
        assert len(ids) == len(set(ids)) == 3
        assert all(i.startswith("agentprobe_call_") for i in ids)


class TestUnsupportedOperationsAreRecorded:
    def test_langchain_stream_is_rejected(self):
        """Before: stream() fell back to a normal call and served the step silently."""
        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"), allow_unused=True)
            with pytest.raises(UnsupportedModelRequest, match="streaming is not supported"):
                list(m.langchain().stream("hi"))
        assert [c.outcome for c in m.calls] == ["unsupported"]
        assert any("streaming is not supported" in f for f in gate_failures(p))

    def test_langchain_astream_is_rejected(self):
        async def consume(llm: Any) -> None:
            async for _ in llm.astream("hi"):
                pass

        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"), allow_unused=True)
            with pytest.raises(UnsupportedModelRequest):
                asyncio.run(consume(m.langchain()))
        assert [c.outcome for c in m.calls] == ["unsupported"]
        assert not p.gate().passed

    def test_langchain_request_without_messages_is_rejected(self):
        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"), allow_unused=True)
            with pytest.raises(UnsupportedModelRequest, match="no messages"):
                m.langchain().invoke([])
        assert not p.gate().passed

    @pytest.mark.parametrize("method", ["HEAD", "OPTIONS", "FOO"])
    def test_http_methods_the_handler_cannot_dispatch(self, method):
        """Before: HEAD got a 501 with no record, and the gate passed."""
        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"), allow_unused=True)
            with m.serve() as s:
                conn = http.client.HTTPConnection(s.host, s.port, timeout=5)
                conn.request(method, "/v1/chat/completions")
                status = conn.getresponse().status
                conn.close()
        assert status == 501
        assert [c.outcome for c in m.calls] == ["unsupported"]
        assert any(f"{method} /v1/chat/completions is not supported" in f for f in gate_failures(p))

    def test_malformed_http_is_recorded(self):
        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"), allow_unused=True)
            with m.serve() as s:
                with socket.create_connection((s.host, s.port), timeout=5) as sock:
                    sock.sendall(b"NOT HTTP AT ALL\r\n\r\n")
                    sock.recv(1024)
        assert [c.outcome for c in m.calls] == ["unsupported"]
        assert not p.gate().passed


class TestInvalidRequestsDoNotConsumeTheScript:
    @pytest.mark.parametrize(
        "body, match",
        [
            ({}, "'model' must be a non-empty string"),
            ({"messages": [{"role": "user", "content": "hi"}]}, "'model'"),
            ({"model": "m", "messages": []}, "'messages' must be a non-empty list"),
            ({"model": "m", "messages": [{"role": "robot", "content": "x"}]}, "role must be"),
            ({"model": "m", "messages": [{"role": "user"}]}, "content must be"),
            (
                {"model": "m", "messages": [{"role": "tool", "content": "r"}]},
                "needs a tool_call_id",
            ),
            (
                {
                    "model": "m",
                    "messages": [
                        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}
                    ],
                },
                "needs an id and a function",
            ),
            ({**USER, "tools": [{"type": "function"}]}, "must be a function tool"),
        ],
    )
    def test_invalid_body_is_rejected_before_a_step_is_chosen(self, body, match):
        """Before: posting {} returned 200, consumed the reply and passed the gate."""
        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"))
            with m.serve() as s:
                status, resp = post(s.base_url, body)
        assert status == 400 and match in resp["error"]["message"]
        assert [c.outcome for c in m.calls] == ["unsupported"]
        failures = gate_failures(p)
        assert any(match in f for f in failures)
        assert any("step 1" in f and "unused" in f for f in failures)  # not consumed


class TestEvidenceCannotBeRewritten:
    def test_editing_model_calls_leaves_the_record_unchanged(self):
        """Before: result.model_calls shared nested dicts with the model's records."""
        with probe(enforce=False) as p:
            m = p.model(script.tool_call("lookup", {"order_id": "1"}), script.reply("x"))
            r = p.run(lambda q: agent_loop(m.langchain(), q), "original")
        r.model_calls[0].messages[0]["content"] = "rewritten"
        r.model_calls[0].returned_tool_calls[0]["args"]["order_id"] = "rewritten"
        fresh = m.calls[0]
        assert fresh.messages[0]["content"] == "original"
        assert fresh.returned_tool_calls[0]["args"] == {"order_id": "1"}

    def test_editing_tool_call_snapshots_leaves_the_record_unchanged(self):
        """The same gap existed for tool calls (result.calls)."""
        with probe(enforce=False) as p:
            tools = p.wrap({"f": lambda x: "ok"})
            tools["f"]({"k": [1]})
            r = p.run(lambda q: "x", "q")
        r.calls[0].args[0]["k"].append("rewritten")
        assert p.calls[0].args == [{"k": [1]}]


class TestMalformedRequestsAreAlwaysRecorded:
    """Review of efa7765: these requests crashed the handler before anything was
    recorded, so a following valid request consumed the script and the gate passed.
    Each test sends the malformed request, then a valid one, and checks the gate
    still fails."""

    @staticmethod
    def _raw(s: Any, data: bytes) -> bytes:
        with socket.create_connection((s.host, s.port), timeout=5) as sock:
            sock.sendall(data)
            try:
                return sock.recv(200).split(b"\r\n")[0]
            except OSError:
                return b""

    @staticmethod
    def _post_bytes(body: bytes, length: str | None = None) -> bytes:
        size = str(len(body)) if length is None else length
        return (
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            b"Content-Type: application/json\r\nConnection: close\r\n"
            + b"Content-Length: "
            + size.encode()
            + b"\r\n\r\n"
            + body
        )

    def _check(self, request: bytes, match: str) -> None:
        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"))
            with m.serve() as s:
                first = self._raw(s, request)
                status, _ = post(s.base_url, USER)  # the script is then used normally
        assert first.startswith(b"HTTP/1.1 400"), first
        assert status == 200
        assert [c.outcome for c in m.calls] == ["unsupported", "served"]
        failures = gate_failures(p)
        assert not p.gate().passed
        assert any(match in f for f in failures), failures

    @pytest.mark.parametrize(
        "body, match",
        [
            ({"model": "m", "messages": [{"role": [], "content": "x"}]}, "role must be"),
            ({"model": "m", "messages": [{"role": {"a": 1}, "content": "x"}]}, "role must be"),
            ({**USER, "tools": 1}, "'tools' must be a list"),
            ({**USER, "tools": "lookup"}, "'tools' must be a list"),
            ({**USER, "tools": [1]}, "must be a function tool"),
            ({"model": "m", "messages": [1]}, "must be an object"),
            (
                {
                    "model": "m",
                    "messages": [{"role": "assistant", "content": None, "tool_calls": 1}],
                },
                "tool_calls must be a non-empty list",
            ),
            ({"model": "m", "messages": [{"role": "user", "content": 5}]}, "content must be"),
            ({"model": ["m"], "messages": [{"role": "user", "content": "x"}]}, "'model'"),
            ([1, 2], "must be a JSON object"),
        ],
    )
    def test_wrongly_typed_json(self, body, match):
        self._check(self._post_bytes(json.dumps(body).encode()), match)

    @pytest.mark.parametrize(
        "length, match",
        [("oops", "invalid Content-Length"), ("-5", "invalid Content-Length"), ("1e3", "invalid")],
    )
    def test_bad_content_length(self, length, match):
        self._check(self._post_bytes(b"{}", length=length), match)

    def test_body_too_large(self, monkeypatch):
        from agentprobe import _scripted_server

        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"))
            with m.serve() as s:
                monkeypatch.setattr(_scripted_server, "_MAX_BODY", 10)
                first = self._raw(s, self._post_bytes(json.dumps(USER).encode()))
                monkeypatch.undo()
                status, _ = post(s.base_url, USER)
        assert first.startswith(b"HTTP/1.1 400") and status == 200
        assert [c.outcome for c in m.calls] == ["unsupported", "served"]
        assert any("too large" in f for f in gate_failures(p))

    def test_client_stalls_mid_body(self, monkeypatch):
        from agentprobe import _scripted_server

        monkeypatch.setattr(_scripted_server, "_SOCKET_TIMEOUT", 0.3)
        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"))
            with m.serve() as s:
                with socket.create_connection((s.host, s.port), timeout=5) as sock:
                    sock.sendall(self._post_bytes(b'{"model"', length="500"))
                    sock.recv(200)  # the server gives up after the socket timeout
                status, _ = post(s.base_url, USER)
        assert status == 200
        assert [c.outcome for c in m.calls] == ["unsupported", "served"]
        assert not p.gate().passed

    @pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "GET"])
    def test_unsupported_method_with_stalled_body(self, monkeypatch, method):
        """Review of 13e1a52: PUT/PATCH bodies were read before the rejection was
        recorded, so a stalled body left no record and the gate passed."""
        from agentprobe import _scripted_server

        monkeypatch.setattr(_scripted_server, "_SOCKET_TIMEOUT", 0.3)
        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"))
            with m.serve() as s:
                first = self._raw(
                    s,
                    f"{method} /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                    "Content-Length: 500\r\n\r\n{".encode(),
                )
                status, _ = post(s.base_url, USER)
        assert first.startswith(b"HTTP/1.1 400"), first  # answered at once, body unread
        assert status == 200
        assert [c.outcome for c in m.calls] == ["unsupported", "served"]
        assert any(f"{method} /v1/chat/completions is not supported" in f for f in gate_failures(p))

    def test_any_unexpected_error_is_recorded(self, monkeypatch):
        """The catch-all: a bug in request handling becomes a recorded rejection,
        never a lost request."""
        from agentprobe import _scripted_server

        def broken(body: Any) -> str | None:
            raise KeyError("simulated handler bug")

        monkeypatch.setattr(_scripted_server, "_invalid", broken)
        with probe(enforce=False) as p:
            m = p.model(script.reply("hi"))
            with m.serve() as s:
                status, resp = post(s.base_url, USER)
                monkeypatch.undo()
                ok, _ = post(s.base_url, USER)
        assert status == 400 and "could not be processed (KeyError" in resp["error"]["message"]
        assert ok == 200
        assert [c.outcome for c in m.calls] == ["unsupported", "served"]
        assert not p.gate().passed
