"""Reports conceal sensitive data without changing execution or evidence."""

import json
import re

import pytest

from agentprobe._redaction import Redactor
from agentprobe.testing import ProbeGateError, probe

SECRETS = (
    "sk-" + "a" * 30,
    "ghp_" + "b" * 36,
    "AKIA" + "A" * 16,
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signature",
    "alice@example.com",
    "opaque-bearer-value",
)
TEXT = " ".join(SECRETS[:-1]) + " Bearer " + SECRETS[-1]


def assert_clean(text):
    for secret in SECRETS:
        assert secret not in text
    assert "[REDACTED:" in text


@pytest.mark.parametrize("redact", [True, False])
def test_reports_and_execution(redact):
    payload = {"nested": [{"api_key": "opaque-key", "text": TEXT}], "safe": 42}
    seen = []

    def tool(data, *, password):
        seen.append((data, password))
        return data

    with probe(enforce=False, redact=redact) as p:
        wrapped = p.wrap(tool)
        result = p.run(lambda _: (wrapped(payload, password="opaque-password"), TEXT)[1], payload)
        result.assert_called_with("tool", payload, password="opaque-password")
        result.assert_answer_contains(SECRETS[0])
        result.assert_answer(lambda answer: answer == TEXT)
        with pytest.raises(AssertionError) as error:
            result.assert_answer_contains("missing")
        surfaces = [
            result.summary(),
            json.dumps(result.to_dict()),
            str(error.value),
            json.dumps(result.calls[0].to_dict()),
            json.dumps(p.calls[0].to_dict()),
        ]
    assert seen == [(payload, "opaque-password")]
    assert seen[0][0] is payload
    assert result.answer == TEXT
    assert result.calls[0].raw_args[0] is payload
    assert result.calls[0].raw_kwargs["password"] == "opaque-password"
    assert payload["nested"][0]["api_key"] == "opaque-key"
    for surface in surfaces:
        if redact:
            assert_clean(surface)
            assert "opaque-key" not in surface
            assert "opaque-password" not in surface
        else:
            assert SECRETS[0] in surface


@pytest.mark.parametrize(
    "text",
    [
        "password=opaque-value",
        '{"api_key": "opaque-value"}',
        "{'access_token': 'opaque-value'}",
        "Authorization: Bearer opaque-value",
        'clientSecret="opaque-value"',
        "PASSWORD: opaque-value",
    ],
)
def test_sensitive_keys_in_text_and_idempotence(text):
    redactor = Redactor()
    cleaned = redactor.text(text)
    assert "opaque-value" not in cleaned
    assert redactor.text(cleaned) == cleaned


def test_custom_rules_and_isolation():
    with probe(enforce=False, redact=[re.compile(r"customer-\d+")], redact_keys=["pin"]) as p:
        tool = p.wrap(lambda data: data)
        result = p.run(lambda _: tool({"pin": 12345, "text": "customer-123 alice@example.com"}), "")
        encoded = json.dumps(result.to_dict())
        assert "12345" not in encoded
        assert "customer-123" not in encoded
        assert "alice@example.com" not in encoded
    with probe() as p:
        result = p.run(lambda _: "customer-123", "")
        assert "customer-123" in result.summary()


def test_exception_gate_and_truncation():
    def tool():
        raise ValueError("x" * 280 + " " + TEXT)

    with pytest.raises(ProbeGateError) as gate:
        with probe() as p:
            result = p.run(lambda _: p.wrap(tool)(), "")
            with pytest.raises(AssertionError) as error:
                result.assert_no_crash()
            assert_clean(str(error.value))
            assert_clean(json.dumps(result.to_dict()))
            assert "sk-aaa" not in result.calls[0].exception_message
    assert_clean(str(gate.value))


def test_snapshot_redacts_before_truncating_repr():
    class Object:
        def __repr__(self):
            return "x" * 180 + " " + SECRETS[0]

    with probe() as p:
        tool = p.wrap(lambda value: value)
        result = p.run(lambda _: tool(Object()), "")
    assert "sk-aaa" not in result.calls[0].value
    assert "sk-aaa" not in json.dumps(result.calls[0].args)


def test_positional_secret_and_exact_assertions():
    def tool(password):
        return "ok"

    with probe(enforce=False) as p:
        result = p.run(lambda _: p.wrap(tool)("opaque-password"), "")
        result.assert_called_with("tool", "opaque-password")
        with pytest.raises(AssertionError) as error:
            result.assert_called_with("tool", "wrong-password")
        assert "opaque-password" not in result.summary()
        assert "wrong-password" not in str(error.value)
        assert result.calls[0].raw_args == ("opaque-password",)


async def test_async_and_late_calls():
    async def tool(data):
        return data

    async with probe() as p:
        wrapped = p.wrap(tool)
        result = await p.arun(lambda _: wrapped({"token": "hidden"}), "")
    await wrapped({"token": "late-hidden"})
    assert result.calls[0].args == [{"token": "[REDACTED:secret]"}]
    assert p.late_calls[0].args == [{"token": "[REDACTED:secret]"}]
    assert p.late_calls[0].raw_args == ({"token": "late-hidden"},)


def test_non_sensitive_data_unchanged():
    redactor = Redactor()
    value = {"count": 3, "items": ["hello", None, True], "url": "https://example.org"}
    assert redactor.value(value) == value


@pytest.mark.parametrize("kwargs", [{"redact": "abc"}, {"redact_keys": "pin"}, {"redact": ["["]}])
def test_invalid_configuration(kwargs):
    with pytest.raises((TypeError, re.error)):
        probe(**kwargs)


def test_nested_sensitive_container_return_value():
    value = {"token": ["first-hidden", {"nested": "second-hidden"}], "safe": "ok"}
    with probe() as p:
        result = p.run(lambda _: p.wrap(lambda: value)(), "")
    assert "first-hidden" not in result.calls[0].value
    assert "second-hidden" not in result.calls[0].value
    assert result.raw_output is value
    assert "first-hidden" not in result.summary()
    assert "second-hidden" not in json.dumps(result.to_dict())


def test_injected_exception_snapshot():
    with probe() as p:
        tool = p.wrap(lambda: "ok")
        p.on("<lambda>").fail_first(1, with_=ValueError(TEXT))
        result = p.run(lambda _: tool(), "")
    assert_clean(json.dumps(result.calls[0].to_dict()))


def test_nested_key_in_free_text():
    text = """response: {"token": ["hidden-one", {"other": "hidden-two"}], "safe": 42}"""
    cleaned = Redactor().text(text)
    assert "hidden-one" not in cleaned
    assert "hidden-two" not in cleaned
    assert '"safe": 42' in cleaned
