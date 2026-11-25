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
        role = str(m.get("role", "unknown"))
        ids = [
            str(c.get("id"))
            for c in m.get("tool_calls") or []
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


def _tool_names(raw: Any) -> list[str]:
    names = []
    for t in raw or []:
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

        def log_message(self, format: str, *args: Any) -> None:  # silence stderr
            pass

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                return json.loads(raw or b"null"), None
            except ValueError:
                return None, "the request body is not valid JSON"

        def _unsupported(self, detail: str, body: Any = None) -> None:
            msgs = _normalize(body.get("messages")) if isinstance(body, dict) else []
            tools = _tool_names(body.get("tools")) if isinstance(body, dict) else []
            self._reject(script._reject_unsupported(msgs, tools, "server", detail))

        def do_POST(self) -> None:
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
