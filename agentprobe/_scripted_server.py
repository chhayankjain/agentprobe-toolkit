"""Local OpenAI Chat Completions endpoint for a scripted model (standard library only).

Get it with ``model.serve()``. One documented surface:
``POST {base_url}/chat/completions`` with ``stream`` absent or false. Anything else
gets HTTP 400 with an ``agentprobe_unsupported`` error and fails the probe's gate.
Rejections use 400, which the OpenAI SDK does not retry; a client that retries
anyway is counted on every attempt, so it can't get around ``max_calls``.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from agentprobe.scripted_model import ModelCallRejected, Served, message

if TYPE_CHECKING:
    from agentprobe.scripted_model import ScriptedModel

_PATH = "/v1/chat/completions"
_MAX_BODY = 10 * 1024 * 1024  # bytes; larger requests are rejected unread
_SOCKET_TIMEOUT = 10.0  # seconds a client may stall mid-request


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return "" if content is None else str(content)


def _normalize(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        role = role if isinstance(role, str) else "unknown"
        calls = m.get("tool_calls")
        ids = [
            str(c.get("id"))
            for c in (calls if isinstance(calls, list) else [])
            if isinstance(c, dict) and c.get("id")
        ]
        out.append(
            message(
                role,
                _text(m.get("content")),
                name=m.get("name") if isinstance(m.get("name"), str) else None,
                tool_call_id=m.get("tool_call_id")
                if isinstance(m.get("tool_call_id"), str)
                else None,
                tool_call_ids=ids,
            )
        )
    return out


_ROLES = {"system", "developer", "user", "assistant", "tool"}


def _invalid(body: dict[str, Any]) -> str | None:
    """Why *body* isn't a valid Chat Completions request, or None. Checked before a
    step is chosen, so a broken client integration can't consume the script."""
    if not isinstance(body.get("model"), str) or not body["model"]:
        return "invalid request: 'model' must be a non-empty string"
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return "invalid request: 'messages' must be a non-empty list"
    for i, m in enumerate(messages):
        where = f"invalid request: messages[{i}]"
        if not isinstance(m, dict):
            return f"{where} must be an object"
        role = m.get("role")
        if not isinstance(role, str) or role not in _ROLES:
            return f"{where}.role must be one of {sorted(_ROLES)}, got {role!r}"
        content = m.get("content")
        calls = m.get("tool_calls")
        if role == "assistant" and calls is not None:
            if not isinstance(calls, list) or not calls:
                return f"{where}.tool_calls must be a non-empty list"
            for j, c in enumerate(calls):
                fn = c.get("function") if isinstance(c, dict) else None
                if (
                    not isinstance(c, dict)
                    or not isinstance(c.get("id"), str)
                    or not isinstance(fn, dict)
                    or not isinstance(fn.get("name"), str)
                    or not isinstance(fn.get("arguments"), str)
                ):
                    return (
                        f"{where}.tool_calls[{j}] needs an id and a function with a name "
                        "and an arguments string"
                    )
            if content is not None and not isinstance(content, (str, list)):
                return f"{where}.content must be a string, a list of parts or null"
        elif not isinstance(content, (str, list)):
            return f"{where}.content must be a string or a list of parts"
        if role == "tool" and not isinstance(m.get("tool_call_id"), str):
            return f"{where} (a tool message) needs a tool_call_id string"
    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            return "invalid request: 'tools' must be a list"
        for i, t in enumerate(tools):
            fn = t.get("function") if isinstance(t, dict) else None
            if (
                not isinstance(t, dict)
                or t.get("type") != "function"
                or not isinstance(fn, dict)
                or not isinstance(fn.get("name"), str)
            ):
                return f"invalid request: tools[{i}] must be a function tool with a name"
    return None


def _tool_names(raw: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(raw, list):
        return names
    for t in raw:
        if isinstance(t, dict) and isinstance(t.get("function"), dict):
            name = t["function"].get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def _completion(served: Served, index: int, model: str) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": served.content}
    if served.tool_calls:
        msg["content"] = served.content or None
        msg["tool_calls"] = [
            {
                "id": cid,
                "type": "function",
                "function": {
                    "name": call.name,
                    # raw_args is delivered byte-for-byte; args as compact JSON.
                    "arguments": call.raw_args
                    if call.raw_args is not None
                    else json.dumps(call.args),
                },
            }
            for cid, call in served.tool_calls
        ]
    return {
        "id": f"chatcmpl-agentprobe-{index}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": "tool_calls" if served.tool_calls else "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class ScriptedServer:
    """A running endpoint. ``base_url`` ends in ``/v1``; ``api_key`` is a dummy
    value for clients that insist on one. Stops on ``close()``, at the end of a
    ``with`` block, or when the probe closes."""

    api_key = "agentprobe-scripted"

    def __init__(self, script: ScriptedModel) -> None:
        self._script = script
        handler = _handler_for(script)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._httpd.daemon_threads = True
        host, port = self._httpd.server_address[:2]
        self.host = str(host)
        self.port = int(port)
        self.base_url = f"http://{self.host}:{self.port}/v1"
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="agentprobe-scripted-server",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> ScriptedServer:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _handler_for(script: ScriptedModel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        #: Socket timeout (seconds): a client that stops sending mid-request is
        #: recorded as unsupported instead of holding the handler forever.
        timeout = _SOCKET_TIMEOUT

        def log_message(self, format: str, *args: Any) -> None:  # silence stderr
            pass

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self._responded = True
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_error(
            self, code: int, message: str | None = None, explain: str | None = None
        ) -> None:
            # Every request the handler can't dispatch (HEAD, OPTIONS, unknown methods,
            # malformed HTTP) ends here: record it so it fails the gate.
            command = getattr(self, "command", None) or "?"
            path = getattr(self, "path", None) or "?"
            script._reject_unsupported(
                [],
                [],
                "server",
                f"{command} {path} is not supported (HTTP {code}: {message or ''}); the "
                f"scripted server implements only POST {_PATH} (OpenAI Chat Completions)",
            )
            super().send_error(code, message, explain)

        def _reject(self, exc: ModelCallRejected) -> None:
            self._send(
                400,
                {
                    "error": {
                        "message": str(exc),
                        "type": f"agentprobe_{exc.outcome}",
                        "param": None,
                        "code": exc.outcome,
                    }
                },
            )

        def _read_json(self) -> tuple[Any, str | None]:
            header = self.headers.get("Content-Length") or "0"
            if not header.strip().isdigit():
                self.close_connection = True  # the body's extent is unknown
                return None, f"invalid Content-Length header {header!r}"
            length = int(header)
            if length > _MAX_BODY:
                self.close_connection = True
                return None, f"request body too large ({length} bytes; limit {_MAX_BODY})"
            raw = self.rfile.read(length) if length > 0 else b""
            if len(raw) < length:
                self.close_connection = True
                return None, f"request body ended after {len(raw)} of {length} bytes"
            try:
                return json.loads(raw or b"null"), None
            except ValueError:
                return None, "the request body is not valid JSON"

        def _unsupported(self, detail: str, body: Any = None) -> None:
            # Recording must succeed whatever the payload looks like.
            try:
                msgs = _normalize(body.get("messages")) if isinstance(body, dict) else []
                tools = _tool_names(body.get("tools")) if isinstance(body, dict) else []
            except Exception:  # noqa: BLE001
                msgs, tools = [], []
            exc = script._reject_unsupported(msgs, tools, "server", detail)  # recorded first
            self._reject(exc)

        def do_POST(self) -> None:
            self._responded = False
            try:
                self._post()
            except Exception as exc:  # noqa: BLE001 - never let a request go unrecorded
                if self._responded:
                    raise  # a response (and its record) already went out
                self.close_connection = True
                self._unsupported(
                    f"the request could not be processed ({type(exc).__name__}: {exc})"
                )

        def _post(self) -> None:
            body, error = self._read_json()
            path = self.path.split("?", 1)[0].rstrip("/")
            if path != _PATH:
                return self._unsupported(
                    f"POST {path} is not supported; the scripted server implements only "
                    f"POST {_PATH} (OpenAI Chat Completions)",
                    body,
                )
            if error is not None or not isinstance(body, dict):
                return self._unsupported(error or "the request body must be a JSON object")
            if body.get("stream"):
                return self._unsupported("streaming is not supported (send stream=false)", body)
            if (problem := _invalid(body)) is not None:
                return self._unsupported(problem, body)
            n = body.get("n", 1)
            if n not in (None, 1):
                return self._unsupported(f"n={n!r} is not supported (one choice only)", body)
            try:
                record, served = script._begin(
                    _normalize(body.get("messages")), _tool_names(body.get("tools")), "server"
                )
            except ModelCallRejected as exc:
                return self._reject(exc)
            try:
                self._send(200, _completion(served, record.index, str(body.get("model") or "")))
            finally:
                script._end()

        def _other(self) -> None:
            body, _ = self._read_json() if self.command in ("PUT", "PATCH") else (None, None)
            self._unsupported(
                f"{self.command} {self.path} is not supported; the scripted server "
                f"implements only POST {_PATH} (OpenAI Chat Completions)",
                body,
            )

        def do_GET(self) -> None:
            self._other()

        def do_PUT(self) -> None:
            self._other()

        def do_PATCH(self) -> None:
            self._other()

        def do_DELETE(self) -> None:
            self._other()

    return Handler


__all__ = ["ScriptedServer"]
