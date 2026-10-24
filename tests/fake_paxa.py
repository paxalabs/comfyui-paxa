"""A local stand-in for the Paxa API, shaped like its published contract.

What it mirrors:
- Errors are RFC 9457 problems, {"title": code, "status": n}, sent as
  application/problem+json, with the status the error catalog gives.
- POST /v1/tts takes text, voice, model, format, stream, speed and language.
  Unstreamed WAV starts with the canonical 44-byte header (16-bit mono PCM,
  24 kHz) with exact sizes; streamed WAV says 0xFFFFFFFF for both sizes.
- POST /v1/stt takes base64 audio and answers text, subtitles and usage. A
  transcription past the server's quiet window answers 200 early, sends
  newlines, and may then carry a problem in the body.
- GET /v1/voices is public; everything else needs the key, as a Bearer token
  or in x-api-key.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

KEY = "fake-key-for-tests"
TTS_RATE = 24_000
MAX_TTS_CHARS = 5_000
STT_MODEL = "paxa-stt-lite-v1-preview"
STT_MAX_BYTES = 26_214_400
STT_MAX_SECONDS = 3_600
LANGUAGE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")

STATUS = {
    "validation": 400,
    "unknown_model": 400,
    "unknown_voice": 400,
    "text_too_long": 400,
    "tag_invalid": 400,
    "audio_invalid": 400,
    "audio_too_long": 400,
    "unauthorized": 401,
    "insufficient_credits": 402,
    "key_limit": 403,
    "not_found": 404,
    "idempotency_in_flight": 409,
    "idempotency_refunded": 409,
    "audio_too_large": 413,
    "idempotency_mismatch": 422,
    "content_blocked": 422,
    "unspeakable_text": 422,
    "rate_limited": 429,
    "concurrency_limited": 429,
    "internal": 500,
    "provider_error": 502,
    "provider_unavailable": 503,
}


def problem_body(code: str) -> bytes:
    return json.dumps({"title": code, "status": STATUS[code]}, separators=(",", ":")).encode()


def wav_bytes(pcm: bytes, rate: int = TTS_RATE, channels: int = 1, streamed: bool = False) -> bytes:
    """The platform's canonical header: RIFF, a 16-byte fmt body, data."""
    size = len(pcm)
    riff = 0xFFFFFFFF if streamed else 36 + size
    data = 0xFFFFFFFF if streamed else size
    header = (
        b"RIFF"
        + struct.pack("<I", riff)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * 2 * channels, 2 * channels, 16)
        + b"data"
        + struct.pack("<I", data)
    )
    return header + pcm


def tone(seconds: float, rate: int = TTS_RATE) -> bytes:
    frames = max(1, int(seconds * rate))
    return b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * n / rate))) for n in range(frames)
    )


@dataclass
class Reply:
    """A scripted answer: sent instead of the normal one, once."""

    status: int = 200
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    drip: int = 0  # newlines sent before the body, after a committed 200
    cut: int | None = None  # send only this many body bytes, then hang up


def problem(code: str, **headers: str) -> Reply:
    return Reply(
        STATUS[code],
        problem_body(code),
        {"content-type": "application/problem+json", **headers},
    )


@dataclass
class Call:
    method: str
    path: str
    headers: dict[str, str]
    body: Any


class FakePaxa:
    def __init__(self, voices: list[dict]) -> None:
        self.voices = voices
        self.calls: list[Call] = []
        self.scripted: dict[str, list[Reply]] = {}
        self.transcript = "สวัสดีค่ะ ยินดีต้อนรับ"
        self._lock = threading.Lock()
        self._counter = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                fake._handle(self, "GET")

            def do_POST(self) -> None:
                fake._handle(self, "POST")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> FakePaxa:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def script(self, method_path: str, *replies: Reply) -> None:
        self.scripted.setdefault(method_path, []).extend(replies)

    def calls_to(self, path: str) -> list[Call]:
        return [call for call in self.calls if call.path == path]

    # Request handling.

    def _handle(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        length = int(handler.headers.get("content-length") or 0)
        raw = handler.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = raw
        headers = {name.lower(): value for name, value in handler.headers.items()}
        with self._lock:
            self.calls.append(Call(method, handler.path, headers, body))
            self._counter += 1
            request_id = f"req_{self._counter:04d}"
            queue = self.scripted.get(f"{method} {handler.path}")
            reply = queue.pop(0) if queue else None
        if reply is None:
            reply = self._answer(method, handler.path, headers, raw, body)
        self._send(handler, reply, request_id)

    def _answer(
        self, method: str, path: str, headers: dict[str, str], raw: bytes, body: Any
    ) -> Reply:
        if method == "GET" and path == "/v1/voices":
            payload = json.dumps({"voices": self.voices}).encode()
            return Reply(200, payload, {"content-type": "application/json"})
        if not (method == "POST" and path in ("/v1/tts", "/v1/stt")):
            return problem("not_found")
        bearer = headers.get("authorization", "")
        if bearer != f"Bearer {KEY}" and not (not bearer and headers.get("x-api-key") == KEY):
            return problem("unauthorized")
        if not isinstance(body, dict):
            return problem("validation")
        return self._tts(body) if path == "/v1/tts" else self._stt(body)

    def _tts(self, body: dict) -> Reply:
        text, voice, model = body.get("text"), body.get("voice"), body.get("model")
        fmt = body.get("format", "mp3")
        speed = body.get("speed", 1)
        if not (
            isinstance(text, str)
            and text
            and isinstance(voice, str)
            and 1 <= len(voice) <= 100
            and isinstance(model, str)
            and 1 <= len(model) <= 100
            and fmt in ("mp3", "opus", "wav")
            and isinstance(body.get("stream", False), bool)
            and isinstance(speed, (int, float))
            and not isinstance(speed, bool)
            and 0.5 <= speed <= 1.5
            and body.get("language", "auto") in ("auto", "th", "en", "zh")
        ):
            return problem("validation")
        roster = {v["id"] for v in self.voices if v["model"] == model}
        if not roster:
            return problem("unknown_model")
        if voice.lower() not in roster:
            return problem("unknown_voice")
        if len(text) > MAX_TTS_CHARS:
            return problem("text_too_long")
        credits = max(0.1, math.ceil(len(text) * 10 / 1000 * 100) / 100)
        headers = {"x-credits-charged": f"{credits:g}"}
        if fmt != "wav":
            return Reply(200, b"ID3not-really-mp3", {"content-type": "audio/mpeg", **headers})
        streamed = body.get("stream", False)
        audio = wav_bytes(tone(min(4.0, 0.05 * len(text) + 0.2)), streamed=streamed)
        if streamed:
            headers["connection"] = "close"
        return Reply(200, audio, {"content-type": "audio/wav", **headers})

    def _stt(self, body: dict) -> Reply:
        audio = body.get("audio")
        vocabulary = body.get("vocabulary", [])
        language = body.get("language", "th")
        if not (
            isinstance(audio, str)
            and audio
            and isinstance(body.get("model"), str)
            and isinstance(language, str)
            and LANGUAGE_PATTERN.match(language)
            and len(language) <= 16
            and body.get("subtitles", "srt") in ("srt", "vtt")
            and body.get("convention", "spoken") in ("spoken", "written")
            and isinstance(vocabulary, list)
            and len(vocabulary) <= 50
            and all(isinstance(t, str) and 1 <= len(t) <= 50 for t in vocabulary)
        ):
            return problem("validation")
        if body["model"] != STT_MODEL:
            return problem("unknown_model")
        try:
            data = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return problem("audio_invalid")
        if len(data) > STT_MAX_BYTES:
            return problem("audio_too_large")
        if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return problem("audio_invalid")
        channels, rate = struct.unpack_from("<HI", data, 22)
        seconds = round((len(data) - 44) / (2 * channels * rate), 3)
        if seconds > STT_MAX_SECONDS:
            return problem("audio_too_long")
        document: dict[str, Any] = {"text": self.transcript}
        if body.get("subtitles") == "srt":
            document["subtitles"] = f"1\n00:00:00,000 --> 00:00:01,500\n{self.transcript}\n"
        elif body.get("subtitles") == "vtt":
            document["subtitles"] = f"WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n{self.transcript}\n"
        document["usage"] = {
            "seconds": seconds,
            "credits": max(0.1, round(seconds / 3600 * 500, 2)),
        }
        return Reply(200, json.dumps(document).encode(), {"content-type": "application/json"})

    def _send(self, handler: BaseHTTPRequestHandler, reply: Reply, request_id: str) -> None:
        if reply.delay:
            time.sleep(reply.delay)
        headers = {
            "x-request-id": request_id,
            "x-ratelimit-limit": "10",
            "x-ratelimit-remaining": "9",
            "x-ratelimit-reset": str(int(time.time()) + 60),
            **reply.headers,
        }
        closing = headers.get("connection") == "close" or reply.drip or reply.cut is not None
        if not closing or reply.cut is not None:
            headers["content-length"] = str(len(reply.body))
        else:
            headers["connection"] = "close"
        try:
            handler.send_response(reply.status)
            for name, value in headers.items():
                handler.send_header(name, value)
            handler.end_headers()
            for _ in range(reply.drip):
                handler.wfile.write(b"\n")
                handler.wfile.flush()
            handler.wfile.write(reply.body[: reply.cut])
        except (BrokenPipeError, ConnectionResetError):
            # The client gave up waiting, as the timeout test intends.
            closing = True
        if closing:
            handler.close_connection = True
