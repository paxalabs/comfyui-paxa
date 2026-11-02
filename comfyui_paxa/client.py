"""Talks to the Paxa API for the nodes.

The API key comes from the environment or a key file, never from a node
input, and never appears in a log line or an error message.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from . import __version__

DEFAULT_BASE_URL = "https://api.paxalabs.com"
BASE_URL_ENV = "PAXA_BASE_URL"
KEY_ENV = "PAXA_API_KEY"
KEY_FILE = Path(__file__).resolve().parent.parent / "paxa_api_key.txt"
CLIENT_TAG = f"comfyui-paxa/{__version__}"

DEFAULT_MODEL = "paxa-tts-flash-v2"
MODELS = ("paxa-tts-flash-v2", "paxa-tts-flash-v1")
STT_MODEL = "paxa-stt-lite-v1-preview"

# (connect, read) seconds. The connect timeout also bounds each block of the
# upload, so a slow upload takes as long as it needs and a stalled one fails.
# The read timeout is the longest silence between bytes of the answer. A
# transcription that runs past 110 s keeps its connection alive by sending a
# newline every 60 s, so 180 s covers it.
TTS_TIMEOUT = (10.0, 120.0)
STT_TIMEOUT = (10.0, 180.0)

MAX_ATTEMPTS = 4
MAX_RETRY_AFTER = 60
# After a lost connection, the first attempt can still be running at Paxa,
# which holds its idempotency key for at most 5 minutes. Asking every 10 s
# stays inside the free plan's 10 requests a minute.
IN_FLIGHT_POLL = 10
IN_FLIGHT_POLLS = 30

log = logging.getLogger("comfyui-paxa")

KEYS_URL = "https://paxalabs.com/app/keys"
PRICING_URL = "https://paxalabs.com/pricing"

MISSING_KEY = (
    "No Paxa API key found. Set the PAXA_API_KEY environment variable before starting "
    "ComfyUI, or save the key in {path}. Get a key at {keys}. "
    "Don't type the key into a node: saved workflows, images and audio carry every node input."
)

MESSAGES = {
    "unauthorized": (
        "Paxa didn't accept the API key. Check the key in PAXA_API_KEY or paxa_api_key.txt. "
        f"Paxa keys start with pxa_ and are listed at {KEYS_URL}."
    ),
    "insufficient_credits": (
        "Your Paxa account doesn't have enough credits for this request. Nothing was charged. "
        f"Top up at {PRICING_URL}, then run the workflow again."
    ),
    "key_limit": (
        "This API key reached the spending cap set on it. Nothing was charged. "
        f"Raise the cap at {KEYS_URL}, or use another key."
    ),
    "unknown_model": "Paxa doesn't serve a model called '{model}'.",
    "unknown_voice": (
        "The voice '{voice}' isn't available on {model}. {voice_hint}"
        "Hear every voice at https://paxalabs.com/voices."
    ),
    "text_too_long": (
        "The text is {chars} characters, more than one request takes. "
        "Split it into shorter parts and run one node per part."
    ),
    "tag_invalid": (
        "The text has a [tag] this model doesn't accept. Nothing was charged. "
        "Remove or change the bracketed tag."
    ),
    "unspeakable_text": (
        "The text has nothing to speak: only tags, emoji or punctuation. Any charge was refunded."
    ),
    "validation": "Paxa refused the request as invalid. Check the node's inputs.",
    "audio_invalid": "Paxa couldn't read the recording. Nothing was charged.",
    "audio_too_long": (
        "The recording is longer than Paxa transcribes in one request. Nothing was charged. "
        "Trim it with Trim Audio Duration."
    ),
    "audio_too_large": (
        "The recording is larger than Paxa takes in one request. Nothing was charged. "
        "Trim it with Trim Audio Duration."
    ),
    "content_blocked": "Paxa's safety system declined this content. The charge was refunded.",
    "rate_limited": (
        "Your Paxa plan's requests-per-minute limit was reached. Wait a minute and run the "
        f"workflow again, or see the plans at {PRICING_URL}."
    ),
    "concurrency_limited": (
        "Your Paxa account already has as many requests running as its plan allows, perhaps "
        "from another app using the same key. Nothing was charged. Try again when they finish."
    ),
    "provider_error": (
        "Paxa's model failed on this request, and the charge was refunded. Run it again. "
        "If it keeps failing, report the request id below."
    ),
    "provider_unavailable": (
        "Paxa's model isn't available right now. Nothing was charged. Try again in a few minutes."
    ),
    "internal": (
        "Paxa hit an internal error. Run it again. "
        "If it keeps failing, report the request id below."
    ),
    "not_found": "Paxa has no endpoint at {url}. If you set PAXA_BASE_URL, check it.",
    "idempotency_in_flight": (
        "Paxa was still working on an earlier attempt at this request after "
        f"{IN_FLIGHT_POLL * IN_FLIGHT_POLLS // 60} minutes, so the node stopped waiting for it. "
        "Running the workflow again sends a new request, which is charged on its own."
    ),
    "idempotency_refunded": (
        "An earlier attempt at this request failed at Paxa and was refunded, so nothing was "
        "charged. Run the workflow again."
    ),
    "idempotency_mismatch": (
        "Paxa refused a retry of this request, because its idempotency key had been used for "
        "a different request. Run the workflow again."
    ),
}


class PaxaError(Exception):
    """A failure to show the user. The message never carries the API key."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        request_id: str | None = None,
    ) -> None:
        details = ", ".join(
            part
            for part in (
                f"HTTP {status}" if status else "",
                code or "",
                f"request {request_id}" if request_id else "",
            )
            if part
        )
        super().__init__(f"{message} [Paxa: {details}]" if details else message)
        self.code = code
        self.status = status
        self.request_id = request_id


def find_api_key() -> str:
    """Returns the key from PAXA_API_KEY, or else from the key file next to the nodes."""
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    if KEY_FILE.is_file():
        try:
            text = KEY_FILE.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            raise PaxaError(
                f"Can't read {KEY_FILE}. Save it as plain text (UTF-8) holding just the key."
            ) from None
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
        raise PaxaError(f"{KEY_FILE} is empty. Paste your Paxa API key into it.")
    doubled = KEY_FILE.with_name(KEY_FILE.name + ".txt")
    if doubled.is_file():
        raise PaxaError(
            f"Found {doubled.name}. Rename it to {KEY_FILE.name}: Windows hid the extension "
            "and added a second .txt."
        )
    raise PaxaError(MISSING_KEY.format(path=KEY_FILE, keys=KEYS_URL))


def base_url() -> str:
    """Returns the API root: PAXA_BASE_URL if set, else the public API.

    A key only travels over https, or over http to this machine.
    """
    url = os.environ.get(BASE_URL_ENV, "").strip().rstrip("/") or DEFAULT_BASE_URL
    parts = urlsplit(url)
    local = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if parts.scheme != "https" and not (parts.scheme == "http" and local):
        raise PaxaError(f"{BASE_URL_ENV} must be an https:// URL, not {url!r}.")
    return url


def synthesize(
    *, text: str, voice: str, model: str, speed: float, language: str, voice_hint: str = ""
) -> bytes:
    """Speaks the text and returns a complete 16-bit mono WAV file.

    voice_hint is added to the message if Paxa doesn't know the voice.
    """
    body = {
        "text": text,
        "voice": voice,
        "model": model,
        "format": "wav",
        # Unstreamed, the WAV header carries exact sizes.
        "stream": False,
        "speed": round(speed, 2),
        "language": language,
    }
    context = {
        "voice": voice,
        "model": model,
        "chars": f"{len(text):,}",
        "voice_hint": voice_hint,
    }
    response = _post("/v1/tts", body, timeout=TTS_TIMEOUT, context=context)
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("audio/"):
        raise PaxaError(
            f"Paxa answered with {content_type or 'no content type'} instead of audio.",
            status=response.status_code,
            request_id=response.headers.get("x-request-id"),
        )
    log.info(
        "Paxa TTS: %d characters, %s on %s, %s credits (request %s)",
        len(text),
        voice,
        model,
        response.headers.get("x-credits-charged", "?"),
        response.headers.get("x-request-id", "?"),
    )
    return response.content


def transcribe(
    *,
    audio_base64: str,
    language: str,
    subtitles: str,
    convention: str,
    vocabulary: list[str],
) -> dict[str, Any]:
    """Transcribes a base64 recording; returns the response document."""
    body: dict[str, Any] = {
        "audio": audio_base64,
        "model": STT_MODEL,
        "subtitles": subtitles,
        "convention": convention,
    }
    if language != "auto":
        body["language"] = language
    if vocabulary:
        body["vocabulary"] = vocabulary
    response = _post("/v1/stt", body, timeout=STT_TIMEOUT, context={"model": STT_MODEL})
    request_id = response.headers.get("x-request-id")
    try:
        document = response.json()
    except ValueError:
        raise PaxaError(
            "Paxa sent a transcription the node couldn't read.",
            status=response.status_code,
            request_id=request_id,
        ) from None
    # A transcription that runs long answers 200 early and keeps the connection
    # alive; if it then fails, the failure arrives as a problem in the body.
    if isinstance(document, dict) and "title" in document and "text" not in document:
        raise _problem_error(document.get("status"), str(document["title"]), request_id, {})
    if not isinstance(document, dict) or not isinstance(document.get("text"), str):
        raise PaxaError(
            "Paxa sent a transcription without text.",
            status=response.status_code,
            request_id=request_id,
        )
    usage = document.get("usage") or {}
    log.info(
        "Paxa STT: %s s of audio, %s credits (request %s)",
        usage.get("seconds", "?"),
        usage.get("credits", "?"),
        request_id or "?",
    )
    return document


_session: requests.Session | None = None


def _http() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _post(
    path: str,
    body: dict[str, Any],
    *,
    timeout: tuple[float, float],
    context: dict[str, str],
) -> requests.Response:
    """POSTs with retries that can't charge twice.

    Every attempt carries the same Idempotency-Key, so a retry after a lost
    connection replays the first charge instead of making a second one. Only
    refusals that charged nothing are retried: rate and concurrency limits, an
    unavailable model, and a replay of a request still in flight, which waits
    for the first attempt to finish.
    """
    url = base_url() + path
    headers = {
        "Authorization": f"Bearer {find_api_key()}",
        "Content-Type": "application/json",
        "User-Agent": CLIENT_TAG,
        "x-paxa-client": CLIENT_TAG,
        "idempotency-key": f"comfyui-paxa-{uuid.uuid4().hex}",
    }
    payload = json.dumps(body, allow_nan=False).encode()
    failures = polls = 0
    lost = False
    while True:
        try:
            # A file rather than bytes: urllib3 sends bytes in one sendall() that has
            # to finish within the connect timeout, and a file block by block.
            response = _http().post(url, data=io.BytesIO(payload), headers=headers, timeout=timeout)
        except requests.exceptions.SSLError as error:
            raise PaxaError(f"Couldn't make a secure connection to {url}: {error}") from None
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as error:
            failures += 1
            if failures == MAX_ATTEMPTS:
                raise PaxaError(f"Couldn't reach Paxa at {url}: {error}") from None
            log.warning("Paxa: %s; retrying (attempt %d of %d)", error, failures + 1, MAX_ATTEMPTS)
            lost = True
            _wait(2 ** (failures - 1))
            continue
        if response.status_code < 400:
            return response
        code = _problem_code(response)
        # A first attempt that lost its connection may still be running, holding
        # the key and, on a plan that runs one request at a time, the slot. Once
        # it finishes, a retry replays its result instead of charging again.
        if polls < IN_FLIGHT_POLLS and (
            code == "idempotency_in_flight" or (lost and code == "concurrency_limited")
        ):
            polls += 1
            log.warning(
                "Paxa: an earlier attempt is still running; asking again in %d s", IN_FLIGHT_POLL
            )
            _wait(IN_FLIGHT_POLL)
            continue
        failures += 1
        delay = _retry_delay(response, code, failures)
        if delay is None or failures == MAX_ATTEMPTS:
            raise _problem_error(
                response.status_code,
                code,
                response.headers.get("x-request-id"),
                {**context, "url": url},
            )
        log.warning("Paxa: %s; retrying in %d s", code or response.status_code, delay)
        _wait(delay)


def _problem_code(response: requests.Response) -> str | None:
    if "json" not in response.headers.get("content-type", ""):
        return None
    try:
        title = response.json().get("title")
    except (ValueError, AttributeError):
        return None
    return title if isinstance(title, str) else None


def _retry_delay(response: requests.Response, code: str | None, attempt: int) -> int | None:
    """Seconds to wait before retrying a refusal that charged nothing, or None."""
    status = response.status_code
    if status == 429 and code == "rate_limited":
        value = response.headers.get("retry-after", "")
        seconds = int(value) if value.isdigit() else 5
        return min(max(seconds, 1), MAX_RETRY_AFTER)
    if status == 429 or status == 503:
        return 2**attempt
    return None


def _problem_error(
    status: Any, code: str | None, request_id: str | None, context: dict[str, str]
) -> PaxaError:
    status = status if isinstance(status, int) else None
    if code in MESSAGES:
        message = MESSAGES[code].format_map(_Blank(context))
    elif code:
        message = f"Paxa refused the request ({code})."
    else:
        message = f"Paxa answered HTTP {status} without an error code. Try again in a few minutes."
    return PaxaError(message, code=code, status=status, request_id=request_id)


class _Blank(dict):
    def __missing__(self, key: str) -> str:
        return "?"


def _wait(seconds: float) -> None:
    """Sleeps, but gives up at once when the user cancels the run in ComfyUI."""
    deadline = time.monotonic() + seconds
    while True:
        _raise_if_interrupted()
        left = deadline - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.25))


def _raise_if_interrupted() -> None:
    try:
        from comfy import model_management
    except ImportError:
        return
    model_management.throw_exception_if_processing_interrupted()
