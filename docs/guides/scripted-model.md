---
description: Script the model's responses to test what your agent does when the model sends bad tool arguments, calls an unknown tool, loops, or replies with nothing. Includes a model-call budget.
---

# Script the model

Tool faults are half of what breaks agents. The other half starts with the **model**:

- it sends arguments that fail the tool's schema;
- it calls a tool that doesn't exist;
- it keeps calling tools, or never calls one, so the loop never ends;
- it replies with empty text.

A scripted model lets you write a test for each of these. You list the responses, AgentProbe serves them in order, and the probe's gate checks that the script was actually used. *Since 1.6.*

!!! note "What a script tests"
    A script tests your agent's **control flow**: *if* the model responds this way, does the agent handle it? It says nothing about model intelligence, or about whether a given model *will* respond this way. For that you still need runs with a real model.

## Bad tool arguments reach the model

The agent below runs tools in a loop. When the model sends a bad tool call, the agent passes the error back so the model can correct itself. The test fails if the error ends the run instead.

```python
from langchain_core.messages import HumanMessage, ToolMessage

from agentprobe.testing import probe, script


def order_status(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id}: shipped"


def my_agent(llm, question):
    bound = llm.bind_tools([order_status])
    messages = [HumanMessage(question)]
    for _ in range(5):
        ai = bound.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content
        for call in ai.tool_calls:
            try:
                result = order_status(**call["args"])
            except TypeError as exc:  # bad arguments: tell the model
                result = f"error: {exc}"
            messages.append(ToolMessage(result, tool_call_id=call["id"]))
    return "I could not finish."


def test_bad_arguments_reach_the_model():
    with probe() as p:
        model = p.model(
            script.tool_call("order_status", {"order": "42"}),  # wrong argument name
            script.tool_call("order_status", {"order_id": "42"}),
            script.reply("Order 42 has shipped."),
            max_calls=5,
        )
        result = p.run(lambda q: my_agent(model.langchain(), q), "Where is order 42?")

        result.assert_no_crash()
        result.assert_model_received("unexpected keyword", role="tool")
        result.assert_every_tool_call_answered()
        result.assert_answer_contains("shipped")
```

If the agent let the `TypeError` escape, the run would end after the first model call:
- `assert_no_crash` would fail;
- `assert_every_tool_call_answered` would be **unassessed**, because no later request reached the model;
- the gate would report steps 2 and 3 as unused.

## A loop that never ends fails the test

`script.repeat(step)` serves the same response on every call. Pair it with `max_calls` so a runaway loop fails the test quickly instead of running thousands of model calls:

```python
def test_a_model_that_never_stops_calling_tools_is_bounded():
    with probe() as p:
        model = p.model(
            script.repeat(script.tool_call("order_status", {"order_id": "42"})),
            max_calls=20,
        )
        result = p.run(lambda q: my_agent(model.langchain(), q), "Where is order 42?")

        result.assert_no_crash()  # my_agent stops itself after 5 turns
        result.assert_model_calls_at_most(5)
```

Request 21 would be rejected with `ModelBudgetExceeded` and fail the gate, *even if the agent caught the exception*.

## Agents that don't use LangChain

`model.serve()` starts a local endpoint that speaks the OpenAI Chat Completions API. Point any client that accepts a `base_url` at it:

```py
import openai

with probe() as p:
    model = p.model(
        script.tool_call("order_status", raw_args='{"order_id": '),  # malformed JSON
        script.reply("Sorry, I could not check that order."),
    )
    with model.serve() as server:
        client = openai.OpenAI(base_url=server.base_url, api_key=server.api_key)
        result = p.run(lambda q: my_openai_agent(client, q), "Where is order 42?")
    result.assert_no_crash()
    result.assert_every_tool_call_answered()
```

The same works for frameworks built on the OpenAI client (pydantic-ai's `OpenAIChatModel`, for example). The endpoint itself needs only the standard library:

```python
import json
import urllib.request


def ask(base_url, question):
    body = {"model": "any", "messages": [{"role": "user", "content": question}]}
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"]


def test_the_server_needs_no_client_library():
    with probe() as p:
        model = p.model(script.reply("hello"))
        with model.serve() as server:
            result = p.run(lambda q: ask(server.base_url, q), "hi")
        result.assert_answer_contains("hello")
```

The server supports **one** request: `POST {base_url}/chat/completions`, without streaming. A streaming request, another endpoint (such as the Responses API) or `n > 1` gets HTTP 400 and fails the gate. It never gets a made-up answer.

## Steps

| Step | The model returns |
|---|---|
| `script.tool_call(name, args=None, *, raw_args=None, id=None)` | One tool call. `raw_args` is a string delivered unchanged, such as malformed JSON. By default every delivery gets a fresh, unique ID. |
| `script.tool_calls(*calls)` | Several tool calls in one response |
| `script.reply(text)` | A final text answer. `reply("")` is the empty reply. |
| `script.repeat(step, times=None)` | The step `times` times, or on every later call. An unbounded `repeat` must come last and needs `max_calls`. |

With the LangChain front end (`model.langchain()`), a `raw_args` string that isn't a JSON object appears in `AIMessage.invalid_tool_calls`, the same way LangChain's provider integrations report unparsable arguments. The raw string is also kept in `additional_kwargs["tool_calls"]`. With the server it is the `arguments` string, byte for byte.

## What fails the gate

The test fails, at the end of the `with` block, if:

| Rule | Exception raised inside the agent |
|---|---|
| A step was never used, or a `repeat(times=n)` was delivered fewer than n times. Turn this off with `allow_unused=True`. | none |
| The agent asked for a response after the last step | `ScriptExhausted` |
| The agent asked for more than `max_calls` responses | `ModelBudgetExceeded` |
| A request arrived while another was still being served. Scripts are sequential, and concurrent model calls aren't supported. | `ScriptOverlap` |
| A request the front end doesn't support: streaming (`stream`/`astream` through LangChain, `stream: true` over HTTP), a request with no messages, an invalid Chat Completions body, another endpoint or HTTP method | `UnsupportedModelRequest` (HTTP 400, or 501 for methods the server can't dispatch) |

An invalid request is rejected before a step is chosen, so a broken client integration can't use up the script and look like it works. Every request the server receives is recorded, including malformed HTTP, a bad `Content-Length`, a body over 10 MB, a client that stalls for 10 seconds anywhere in a request (request line, headers or body), and any unexpected error while handling it. An idle connection that sends nothing is not a request and is not recorded. The server checks for a `model`, a non-empty `messages` list with known roles, a `tool_call_id` on tool messages, complete assistant tool calls, and function tools with names.

Tool-call IDs are unique within a script. Two steps can't use the same explicit `id`, and explicit IDs can't start with `agentprobe_call_`, the prefix used for generated ones. Otherwise an answer to one call could be mistaken for an answer to another.

Each exception is recorded **before** it is raised, so the gate fails even if your agent catches it. All of them subclass `ModelCallRejected` (an `Exception`). They are harness failures, not simulated provider errors.

The budget is checked, and the response chosen, under one lock before anything is returned. At most `max_calls` requests are ever served. Every attempt counts, including retries: the server rejects with HTTP 400, which the OpenAI SDK doesn't retry, and a client that retries anyway is rejected each time.

Requests that arrive after the probe has closed (for example from an agent still running past its `timeout`) are rejected with `ModelCallAfterProbe` and recorded as late. They are never evidence, and they don't fail the gate. The server stops when the probe closes, even if your test forgot the `with` block. After that, requests fail with "connection refused".

## Assertions

| Assertion | Passes when | Unassessed when |
|---|---|---|
| `assert_model_calls_at_most(n)` | At most *n* requests reached the scripted model, rejected ones included | The agent is still running |
| `assert_model_received(pattern, *, role=None)` | A message sent to the model matches *pattern* (a case-insensitive substring or a compiled regex). `role` is `"user"`, `"assistant"`, `"tool"` or `"system"`. | The run ended right after the model returned tool calls, so nothing after them was observable |
| `assert_every_tool_call_answered()` | Every tool call the model returned came back as a `tool` message with the same ID in a later request, including several calls in one response | A call had no later request, or no tool calls were returned |

**Unassessed is not passed**, and it fails a required assertion. An assertion only **fails** when there is evidence of absence: later requests reached the model and the expected message wasn't in them. `assert_every_tool_call_answered` checks that tool calls were completed. It says nothing about whether the task was done correctly.

`result.model_calls` holds every request, as copies: editing them never changes the recorded evidence. Each record includes the outcome, the step served, the messages received (role, content, name, tool-call IDs), the tools offered and the tool calls returned. The records are also in `result.to_dict()`.

## Limits

- Scripts are sequential. Responses are not matched to requests (replaying recorded responses is planned in [#105](https://github.com/chhayankjain/agentprobe-toolkit/issues/105)).
- No streaming. Through LangChain this includes a LangGraph graph run with `stream_mode="messages"`, which streams the model; `invoke`, and the other stream modes, work.
- Only one API surface for the server.
- One scripted model serves whoever calls it. If your agent uses the same model object for several roles, such as an agent, a checker and a summarizer, all of their calls consume steps in arrival order.
- Faults at the model provider, like rate limits and timeouts, aren't covered yet ([#106](https://github.com/chhayankjain/agentprobe-toolkit/issues/106)).
