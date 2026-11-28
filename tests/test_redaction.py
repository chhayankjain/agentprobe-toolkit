"""Reports conceal sensitive data without changing execution or evidence."""

import datetime
import decimal
import json
import pathlib
import re
import subprocess
import sys
import uuid

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
    assert result.calls[0].args == [{"token": p._redactor.marker("hidden")}]
    assert p.late_calls[0].args == [{"token": p._redactor.marker("late-hidden")}]
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


def test_redacting_megabyte_word_is_fast():
    # A subprocess bounds the test even if quadratic scanning is reintroduced.
    # Time only filtering; interpreter startup is outside the one-second budget.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from time import perf_counter
from agentprobe._redaction import Redactor
redactor = Redactor()
text = "A" * 1_000_000
start = perf_counter()
assert redactor.text(text) == text
elapsed = perf_counter() - start
assert elapsed < 1.0, f"Redacting a 1 MB word took {elapsed:.3f}s"
""",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("text", ["ACGT" * 2500, "Ab09_-" * 1667])
def test_large_tool_result_does_not_fail_timing_assertion(text):
    with probe() as p:
        tool = p.wrap(lambda: text)
        result = p.run(lambda _: tool(), "")
        result.assert_within(1.0)
    assert result.answer == text


@pytest.mark.parametrize("key", ["access_token", "clientSecret", "service-api-key"])
def test_prefixed_sensitive_key_keeps_its_name(key):
    redactor = Redactor()
    text = f'{key}="opaque-value"; safe=ok'
    assert redactor.text(text) == f"{key}={redactor.marker('opaque-value')}; safe=ok"


@pytest.mark.parametrize("redact", [True, False])
@pytest.mark.parametrize("key", ["password", "pin"])
def test_uncalled_tool_positional_secret(redact, key):
    def login(password, /):
        return "ok"

    def unlock(pin, /):
        return "ok"

    with probe(enforce=False, redact=redact, redact_keys=["pin"]) as p:
        p.wrap({"authenticate": login if key == "password" else unlock})
        result = p.run(lambda _: "skipped", "")
        with pytest.raises(AssertionError) as error:
            result.assert_called_with("authenticate", "opaque-demo-credential")
        assert result.calls == []
        assert result.assertions[-1].status == "fail"
        assert not result.gate().passed
        surfaces = [
            str(error.value),
            result.summary(),
            json.dumps(result.to_dict()),
            str(result.gate()),
            str(p.gate()),
        ]
        for surface in surfaces:
            assert ("opaque-demo-credential" in surface) is (not redact)
        assert "no matching invocation" in str(error.value)


def test_uncalled_tool_signature_is_frozen_in_result():
    def login(password):
        return "ok"

    def other(value):
        return value

    with probe(enforce=False) as p:
        p.wrap({"login": login})
        result = p.run(lambda _: "skipped", "")
        # Later registration must not rewrite an earlier result's report metadata.
        p.wrap({"login": other})
        with pytest.raises(AssertionError) as error:
            result.assert_called_with("login", "opaque-demo-credential")
        assert "opaque-demo-credential" not in str(error.value)
        assert "_signatures" not in result.to_dict()


async def test_uncalled_async_tool_positional_secret():
    async def login(password):
        return "ok"

    async with probe(enforce=False) as p:
        p.wrap(login)
        result = await p.arun(lambda _: "skipped", "")
        with pytest.raises(AssertionError) as error:
            result.assert_called_with("login", "opaque-demo-credential")
        assert "opaque-demo-credential" not in str(error.value)


def test_redacted_values_have_per_probe_identity():
    first, second = Redactor(), Redactor()
    alice = first.text("alice@example.com")
    bob = first.text("bob@example.com")
    assert alice != bob
    assert first.text("alice@example.com") == alice
    assert second.text("alice@example.com") != alice
    assert first.text(alice) == alice
    assert first.marker("one") != first.marker("two")
    assert first.marker("one") == first.marker("one")


def test_failure_message_distinguishes_expected_and_actual_email():
    with probe(enforce=False) as p:
        result = p.run(lambda _: "Contact bob@example.com", "")
        with pytest.raises(AssertionError) as error:
            result.assert_answer_contains("alice@example.com")
        expected = p._redactor.text("alice@example.com")
        actual = p._redactor.text("bob@example.com")
        assert expected != actual
        assert expected in str(error.value)
        assert actual in str(error.value)
        assert "alice@example.com" not in str(error.value)
        assert "bob@example.com" not in str(error.value)
        assert result.to_dict() == result.to_dict()


@pytest.mark.parametrize("redact", [True, False])
def test_redacted_dictionary_keys_preserve_entries_on_every_surface(redact):
    value = {"alice@example.com": 1, "bob@example.com": 2}
    with probe(redact=redact) as p:
        tool = p.wrap(lambda data: data)
        result = p.run(lambda _: tool(value), value)
    snapshots = [
        result.calls[0].args[0],
        p.calls[0].args[0],
        result.to_dict()["input"],
        result.to_dict()["observed"]["calls"][0]["args"][0],
    ]
    for snapshot in snapshots:
        assert len(snapshot) == 2
        assert sorted(snapshot.values()) == [1, 2]
        assert p._redactor.value(snapshot) == snapshot
        if redact:
            assert "alice@example.com" not in json.dumps(snapshot)
        else:
            assert snapshot == value
    assert value == {"alice@example.com": 1, "bob@example.com": 2}


def test_short_digest_collisions_do_not_overwrite_entries(monkeypatch):
    redactor = Redactor()
    monkeypatch.setattr(
        redactor,
        "_digest",
        lambda value: "a" * 16 + ("b" * 48 if value.startswith("alice") else "c" * 48),
    )
    output = redactor.value({"alice@example.com": 1, "bob@example.com": 2})
    assert len(output) == 2
    assert sorted(output.values()) == [1, 2]
    assert redactor.value(output) == output
    assert redactor.text("alice@example.com") in output
    assert redactor.text("bob@example.com") in output


def test_literal_marker_key_cannot_overwrite_redacted_key():
    from agentprobe.probe import _jsonable

    redactor = Redactor()
    marker = redactor.text("alice@example.com")
    original = {"alice@example.com": 1, marker: 2, marker + "~2": 3}
    for output in [redactor.value(original), _jsonable(original, redactor=redactor)]:
        assert len(output) == 3
        assert sorted(output.values()) == [1, 2, 3]
        assert redactor.value(output) == output
        assert "alice@example.com" not in output


def test_custom_patterns_do_not_rewrite_generated_markers():
    redactor = Redactor([r"\d+", r"REDACTED"])
    output = redactor.value({"token": "123", "text": "123 alice@example.com"})
    assert redactor.value(output) == output


def test_identity_is_shared_across_concurrent_calls():
    from concurrent.futures import ThreadPoolExecutor

    redactor = Redactor()
    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(redactor.text, ["alice@example.com"] * 40))
    assert len(set(values)) == 1


def test_sensitive_opaque_value_with_broken_repr_still_reaches_tool():
    class Opaque:
        def __repr__(self):
            raise ValueError("cannot render")

    secret = Opaque()
    with probe() as p:

        def login(password):
            assert password is secret
            return "ok"

        result = p.run(lambda _: p.wrap(login)(secret), "")
        result.assert_no_crash()
    assert result.calls[0].raw_args[0] is secret
    assert result.calls[0].args[0].startswith("[REDACTED:secret#")


def test_secret_identity_matches_structured_and_text_fields():
    redactor = Redactor()
    marker = redactor.value({"password": "opaque-value"})["password"]
    assert redactor.text('password="opaque-value"') == "password=" + marker
    assert redactor.text("password=" + marker) == "password=" + marker


@pytest.mark.parametrize("redact", [True, False])
def test_return_scenario_redacts_before_shortening(redact):
    value = "x" * 48 + " " + SECRETS[0]
    with probe(enforce=False, redact=redact) as p:
        tool = p.wrap(lambda: "original")
        p.on("<lambda>").returns(value, times=2)
        result = p.run(lambda _: tool(), "")
    assert result.answer == value
    for report in [result.summary(), json.dumps(result.to_dict()), str(p.gate())]:
        assert ("sk-aaa" in report) is (not redact)


@pytest.mark.parametrize("kind", ["reply", "args", "raw_args"])
@pytest.mark.parametrize("redact", [True, False])
def test_script_description_redacts_before_shortening(kind, redact):
    from agentprobe.testing import script

    value = "x" * (28 if kind == "reply" else 38) + " " + SECRETS[0]
    step = (
        script.reply(value)
        if kind == "reply"
        else script.tool_call(
            "lookup",
            **({kind: {"q": value}} if kind == "args" else {kind: json.dumps({"q": value})}),
        )
    )
    with probe(enforce=False, redact=redact) as p:
        model = p.model(script.repeat(step, times=2))

        def agent(_):
            _, served = model._begin([], ["lookup"], "test")
            model._end()
            if kind == "reply":
                assert served.content == value
            else:
                call = served.tool_calls[0][1]
                assert (call.args if kind == "args" else json.loads(call.raw_args)) == {"q": value}
            return "done"

        result = p.run(agent, "")
    for report in [result.summary(), json.dumps(result.to_dict()), str(p.gate())]:
        assert ("sk-aaa" in report) is (not redact)


def test_unused_descriptions_use_probe_custom_patterns_without_mutating_script():
    from agentprobe.testing import script

    value = "x" * 26 + " customer-12345678901234567890"
    step = script.reply(value)
    with probe(enforce=False, redact=[r"customer-\d+"]) as p:
        p.model(step)
        p.on("lookup").returns(value)
        result = p.run(lambda _: "done", "")
    for report in [result.summary(), json.dumps(result.to_dict()), str(p.gate())]:
        assert "customer-" not in report
    assert step.text == value
    assert "customer-" in step.describe()
    with probe(enforce=False, redact=False) as p:
        p.model(step)
        assert "customer-" in str(p.gate())


@pytest.mark.parametrize("redact", [True, False])
@pytest.mark.parametrize(
    "value",
    [
        pathlib.PurePath("/tmp/alice@example.com"),
        uuid.UUID("12345678-1234-5678-1234-567812345678"),
        decimal.Decimal("12345.6789"),
        datetime.date(2026, 11, 28),
        datetime.time(12, 34, 56),
        datetime.datetime(2026, 11, 28, 12, 34, 56),
    ],
)
def test_converted_values_redacted_in_standalone_call_exports(value, redact):
    text = value.isoformat() if isinstance(value, (datetime.date, datetime.time)) else str(value)
    rules = [re.escape(text)] if redact else False
    payload = {"nested": [value]}
    with probe(redact=rules) as p:

        def tool(arg, *, data):
            assert arg is value
            assert data is payload
            return "ok"

        wrapped = p.wrap(tool)
        result = p.run(lambda _: wrapped(value, data=payload), "")
        result.assert_called_with("tool", value, data=payload)
    wrapped(value, data=payload)
    for call in [result.calls[0], p.calls[0], p.late_calls[0]]:
        assert call.raw_args[0] is value
        assert call.raw_kwargs["data"] is payload
        exported = json.dumps(call.to_dict())
        assert (text in exported) is (not redact)
        assert call.args[0] == call.kwargs["data"]["nested"][0]
    assert (text in json.dumps(result.to_dict())) is (not redact)


def test_path_email_uses_default_redaction_in_call_snapshot():
    value = pathlib.PurePath("/tmp/alice@example.com")
    with probe() as p:
        result = p.run(lambda _: p.wrap(lambda arg: "ok")(value), "")
    assert "alice@example.com" not in json.dumps(result.calls[0].to_dict())
    assert result.calls[0].args[0] == p._redactor.text(str(value))
    assert "[REDACTED:email#" in result.calls[0].args[0]


def test_non_sensitive_converted_values_keep_their_representation():
    from agentprobe.probe import _jsonable

    values = [
        pathlib.PurePath("/tmp/report.txt"),
        decimal.Decimal("12.34"),
        uuid.UUID("12345678-1234-5678-1234-567812345678"),
        datetime.date(2026, 11, 28),
    ]
    assert _jsonable(values, redactor=Redactor()) == _jsonable(values)


@pytest.mark.parametrize(
    "key",
    [
        "aws_secret_access_key",
        "AWS-SECRET-ACCESS-KEY",
        "Cookie",
        "cookie",
        "private_key",
        "privateKey",
    ],
)
def test_additional_sensitive_field_names(key):
    secret = "opaque-credential-value"
    payload = {"nested": [{key: secret}]}
    with probe(enforce=False) as p:
        tool = p.wrap(lambda data: data)
        result = p.run(lambda _: tool(payload), "")
        result.assert_called_with("<lambda>", payload)
        with pytest.raises(AssertionError) as error:
            result.assert_answer_contains("missing")
    for report in [
        json.dumps(result.calls[0].to_dict()),
        result.summary(),
        json.dumps(result.to_dict()),
        str(error.value),
        Redactor().text(f'{key}="{secret}"'),
    ]:
        assert secret not in report
    assert result.raw_output is payload
    assert result.calls[0].raw_args[0] is payload
    assert Redactor(False).value(payload) == payload


def test_additional_key_names_do_not_match_unrelated_fields():
    payload = {
        "cookie_count": 2,
        "private_key_format": "PEM",
        "aws_secret_access_key_length": 40,
        "public_key": "public-value",
    }
    assert Redactor().value(payload) == payload


PROVIDER_CREDENTIALS = [
    prefix + "A1b2C3d4" * 4
    for prefix in (
        "sk_live_",
        "sk_test_",
        "rk_live_",
        "rk_test_",
        "xoxb-",
        "xoxp-",
        "xoxa-",
        "xoxr-",
        "xoxs-",
        "hf_",
        "tvly-",
    )
] + ["AIza" + "A" * 35]


@pytest.mark.parametrize("credential", PROVIDER_CREDENTIALS)
@pytest.mark.parametrize("redact", [True, False])
def test_provider_credentials_in_reports(credential, redact):
    text = f"Credential ({credential}), request failed"
    with probe(enforce=False, redact=redact) as p:

        def tool(data):
            assert data == {"nested": [text]}
            raise ValueError(text)

        result = p.run(lambda _: p.wrap(tool)({"nested": [text]}), text)
        with pytest.raises(AssertionError) as error:
            result.assert_no_crash()
    for report in [
        result.summary(),
        json.dumps(result.to_dict()),
        json.dumps(result.calls[0].to_dict()),
        str(error.value),
    ]:
        assert (credential in report) is (not redact)
    assert str(result.exception) == text
    redactor = Redactor(redact)
    filtered = redactor.text(text)
    assert redactor.text(filtered) == filtered
    assert filtered.startswith("Credential (") and filtered.endswith("), request failed")


@pytest.mark.parametrize(
    "text",
    [
        "sk_live_",
        "sk_test_example",
        "rk_live_short",
        "AIzaExample",
        "xoxb-short",
        "hf_example",
        "tvly-demo",
        "pk_live_" + "A" * 32,
        "not_hf_" + "A" * 32,
        "xoxo-" + "A" * 32,
    ],
)
def test_provider_patterns_leave_non_credentials_unchanged(text):
    assert Redactor().text(text) == text


@pytest.mark.parametrize("text", ["langchain@1.4.9", "@scope/package@2.0.1", "tool@10.20.30"])
def test_package_versions_are_not_emails(text):
    assert Redactor().text(text) == text


@pytest.mark.parametrize(
    "text", ["alice@example.com", "alice@sub.example.co.uk", "alice@example.technology"]
)
def test_alphabetic_email_domains_remain_redacted(text):
    assert text not in Redactor().text(text)
