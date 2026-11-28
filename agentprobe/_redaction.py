"""Dependency-free report redaction; never used for execution or comparisons."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

Redact = bool | Sequence[str | re.Pattern[str]]
_DEFAULT_KEYS = ("api_key", "token", "password", "secret", "authorization")
_PATTERNS = (
    ("bearer", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.I)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")),
    (
        "api_key",
        re.compile(
            r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"
        ),
    ),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"),
    ),
)


def _normalize(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


class Redactor:
    def __init__(self, redact: Redact = True, keys: Sequence[str] = ()) -> None:
        if isinstance(redact, (str, bytes)) or not isinstance(redact, (bool, Sequence)):
            raise TypeError("redact must be a bool or a sequence of regex patterns")
        if isinstance(keys, (str, bytes)) or any(
            not isinstance(k, str) or not _normalize(k) for k in keys
        ):
            raise TypeError("redact_keys must be a sequence of nonempty strings")
        self.enabled = redact is not False
        self.keys = {_normalize(k) for k in (*_DEFAULT_KEYS, *keys)}
        self.patterns = list(_PATTERNS)
        if not isinstance(redact, bool):
            self.patterns.extend(("custom", re.compile(p)) for p in redact)
        # Match key/value pairs in prose, JSON, and Python reprs, including quoted
        # values containing spaces. Keep the key visible to explain the removal.
        key_pattern = "|".join("[_-]*".join(map(re.escape, k)) for k in sorted(self.keys))
        self.pairs = re.compile(rf"""["']?[\w-]*(?:{key_pattern})["']?\s*[:=]\s*""", re.I)

    def sensitive(self, key: str) -> bool:
        normalized = _normalize(key)
        return self.enabled and any(normalized.endswith(k) for k in self.keys)

    def text(self, text: str) -> str:
        if not self.enabled:
            return text
        parts: list[str] = []
        end = 0
        for match in self.pairs.finditer(text):
            if match.start() < end:
                continue
            parts.append(text[end : match.end()])
            end = _value_end(text, match.end())
            parts.append("[REDACTED:secret]")
        parts.append(text[end:])
        text = "".join(parts)
        for label, pattern in self.patterns:
            text = pattern.sub(lambda _: f"[REDACTED:{label}]", text)
        return text

    def value(self, value: Any) -> Any:
        """Copy JSON-safe report data, filtering values and free-text keys."""
        if not self.enabled:
            return value
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, dict):
            return {
                self.text(k): "[REDACTED:secret]" if self.sensitive(k) else self.value(v)
                for k, v in value.items()
            }
        return value


def _value_end(text: str, start: int) -> int:
    """Find a quoted, balanced container, or bare value without evaluating it.

    An unterminated quoted/container value consumes the rest conservatively.
    This also handles reprs of nested containers in answers and exceptions.
    """
    if text.startswith("[REDACTED:", start):
        close = text.find("]", start)
        return len(text) if close < 0 else close + 1
    bearer = re.match(r"Bearer\s+", text[start:], re.I)
    cursor = start + bearer.end() if bearer else start
    stack: list[str] = []
    quote: str | None = None
    while cursor < len(text):
        char = text[cursor]
        if quote:
            if char == "\\":
                cursor += 2
                continue
            if char == quote:
                quote = None
                if not stack:
                    return cursor + 1
        elif char in "\"'":
            quote = char
        elif char in "[{(":
            stack.append(char)
        elif char in "]})":
            if not stack:
                return cursor
            stack.pop()
            if not stack:
                return cursor + 1
        elif not stack and (char.isspace() or char in ",;"):
            return cursor
        cursor += 1
    return len(text)
