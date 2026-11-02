from __future__ import annotations

import base64
import re
import socket
import struct
import sys
import time
import types

import pytest
from fake_paxa import KEY, STATUS, STT_MAX_BYTES, Reply, Uplink, problem, wav_bytes

from comfyui_paxa import __version__, client

# The platform's own rule for x-paxa-client (src/lib/client-tag.ts).
CLIENT_TAG_RULE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}(?:/[a-z0-9][a-z0-9._+-]{0,23})?$")
IDEMPOTENCY_KEY_RULE = re.compile(r"^[!-~]{1,200}$")


def speak(**overrides):
    args = {
        "text": "สวัสดีค่ะ Hello",
        "voice": "foithong",
        "model": "paxa-tts-flash-v2",
        "speed": 1.0,
        "language": "auto",
    }
    return client.synthesize(**{**args, **overrides})


def wav_base64(seconds: float = 0.5, rate: int = 16_000) -> str:
    return base64.b64encode(wav_bytes(b"\0\0" * int(seconds * rate), rate=rate)).decode()


def transcribe(**overrides):
    args = {
        "audio_base64": wav_base64(),
        "language": "auto",
        "subtitles": "srt",
        "convention": "spoken",
        "vocabulary": [],
    }
    return client.transcribe(**{**args, **overrides})


# The key.


def test_key_from_environment(monkeypatch):
    monkeypatch.setenv("PAXA_API_KEY", "  env-key  ")
    client.KEY_FILE.write_text("file-key")
    assert client.find_api_key() == "env-key"


def test_key_from_file_skips_bom_comments_and_blank_lines():
    client.KEY_FILE.write_text("﻿# my Paxa key\n\n  file-key  \nignored\n", encoding="utf-8")
    assert client.find_api_key() == "file-key"


def test_empty_key_file_is_an_error():
    client.KEY_FILE.write_text("\n# nothing yet\n")
    with pytest.raises(client.PaxaError, match="is empty"):
        client.find_api_key()


def test_key_file_in_utf16_is_explained():
    client.KEY_FILE.write_text("file-key", encoding="utf-16")
    with pytest.raises(client.PaxaError, match=r"plain text \(UTF-8\)"):
        client.find_api_key()


def test_doubled_txt_extension_is_named():
    client.KEY_FILE.with_name("paxa_api_key.txt.txt").write_text("file-key")
    with pytest.raises(client.PaxaError, match=r"Rename it to paxa_api_key\.txt"):
        client.find_api_key()


def test_missing_key_explains_both_places_and_why_not_a_node_input():
    with pytest.raises(client.PaxaError) as error:
        client.find_api_key()
    message = str(error.value)
    assert "PAXA_API_KEY" in message
    assert str(client.KEY_FILE) in message
    assert "Don't type the key into a node" in message


# The base URL.


def test_base_url_defaults_to_the_public_api():
    assert client.base_url() == "https://api.paxalabs.com"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("https://example.com/", "https://example.com"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
        ("http://localhost:8080/", "http://localhost:8080"),
    ],
)
def test_base_url_override(monkeypatch, value, expected):
    monkeypatch.setenv("PAXA_BASE_URL", value)
    assert client.base_url() == expected


@pytest.mark.parametrize(
    "value", ["http://api.paxalabs.com", "ftp://example.com", "api.paxalabs.com"]
)
def test_key_never_goes_over_plain_http_to_another_host(monkeypatch, value):
    monkeypatch.setenv("PAXA_BASE_URL", value)
    with pytest.raises(client.PaxaError, match="must be an https"):
        client.base_url()


# Text to speech.


def test_synthesize_sends_the_contract_request(fake):
    audio = speak(text="Hello", voice="massaman", speed=1.25, language="en")
    (call,) = fake.calls_to("/v1/tts")
    assert call.method == "POST"
    assert call.body == {
        "text": "Hello",
        "voice": "massaman",
        "model": "paxa-tts-flash-v2",
        "format": "wav",
        "stream": False,
        "speed": 1.25,
        "language": "en",
    }
    assert call.headers["authorization"] == f"Bearer {KEY}"
    assert call.headers["content-type"] == "application/json"
    assert call.headers["x-paxa-client"] == f"comfyui-paxa/{__version__}"
    assert call.headers["user-agent"] == f"comfyui-paxa/{__version__}"
    assert IDEMPOTENCY_KEY_RULE.match(call.headers["idempotency-key"])
    assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
    assert struct.unpack_from("<I", audio, 40)[0] == len(audio) - 44


def test_client_tag_passes_the_platform_rule():
    assert CLIENT_TAG_RULE.match(client.CLIENT_TAG)


def test_speed_is_rounded_to_two_decimals(fake):
    speak(speed=0.7000000000000001)
    assert fake.calls_to("/v1/tts")[0].body["speed"] == 0.7


def test_audio_that_isnt_audio_is_an_error(fake):
    fake.script("POST /v1/tts", Reply(200, b'{"ok":true}', {"content-type": "application/json"}))
    with pytest.raises(client.PaxaError, match="instead of audio"):
        speak()


TTS_CODES = [
    "validation",
    "unknown_model",
    "unknown_voice",
    "text_too_long",
    "tag_invalid",
    "unauthorized",
    "insufficient_credits",
    "key_limit",
    "idempotency_refunded",
    "idempotency_mismatch",
    "unspeakable_text",
    "internal",
    "provider_error",
]


@pytest.mark.parametrize("code", TTS_CODES)
def test_refusals_carry_code_status_and_request_id_but_never_the_key(fake, waits, code):
    fake.script("POST /v1/tts", problem(code))
    with pytest.raises(client.PaxaError) as error:
        speak()
    assert error.value.code == code
    assert error.value.status == STATUS[code]
    assert error.value.request_id == "req_0001"
    assert KEY not in str(error.value)
    assert waits == []
    assert len(fake.calls_to("/v1/tts")) == 1


def test_known_codes_have_their_own_message(fake):
    fake.script("POST /v1/tts", problem("insufficient_credits"))
    with pytest.raises(client.PaxaError, match="Nothing was charged"):
        speak()


def test_unknown_voice_message_names_voice_model_and_hint(fake):
    with pytest.raises(client.PaxaError) as error:
        speak(voice="somtam", voice_hint="It speaks on paxa-tts-flash-v1; switch the model. ")
    message = str(error.value)
    assert "'somtam' isn't available on paxa-tts-flash-v2" in message
    assert "switch the model" in message


def test_text_too_long_names_the_length(fake):
    with pytest.raises(client.PaxaError, match="5,001 characters"):
        speak(text="ก" * 5001)


def test_wrong_key_is_unauthorized(fake, monkeypatch):
    monkeypatch.setenv("PAXA_API_KEY", "pxa_wrong")
    with pytest.raises(client.PaxaError) as error:
        speak()
    assert error.value.code == "unauthorized"
    assert "pxa_wrong" not in str(error.value)


def test_error_without_problem_body(fake):
    fake.script(
        "POST /v1/tts", Reply(520, b"<html>origin error</html>", {"content-type": "text/html"})
    )
    with pytest.raises(client.PaxaError, match="without an error code") as error:
        speak()
    assert error.value.status == 520


def test_unknown_code_is_named(fake):
    fake.script(
        "POST /v1/tts",
        Reply(
            400,
            b'{"title":"brand_new_code","status":400}',
            {"content-type": "application/problem+json"},
        ),
    )
    with pytest.raises(client.PaxaError, match=r"Paxa refused the request \(brand_new_code\)"):
        speak()


# Retries: only refusals that charged nothing, and always with the same
# Idempotency-Key, so a retry can't charge twice.


def test_rate_limit_waits_for_retry_after_then_succeeds(fake, waits):
    fake.script("POST /v1/tts", problem("rate_limited", **{"retry-after": "7"}))
    assert speak()[:4] == b"RIFF"
    assert waits == [7]
    first, second = fake.calls_to("/v1/tts")
    assert first.headers["idempotency-key"] == second.headers["idempotency-key"]


@pytest.mark.parametrize("retry_after, expected", [("600", 60), ("0", 1), ("soon", 5)])
def test_retry_after_is_bounded(fake, waits, retry_after, expected):
    fake.script("POST /v1/tts", problem("rate_limited", **{"retry-after": retry_after}))
    speak()
    assert waits == [expected]


@pytest.mark.parametrize(
    "code", ["concurrency_limited", "provider_unavailable", "idempotency_in_flight"]
)
def test_refusals_that_charged_nothing_back_off(fake, waits, code):
    fake.script("POST /v1/tts", problem(code), problem(code))
    speak()
    assert waits == [2, 4]
    keys = {call.headers["idempotency-key"] for call in fake.calls_to("/v1/tts")}
    assert len(keys) == 1


def test_retries_stop_after_four_attempts(fake, waits):
    fake.script("POST /v1/tts", *[problem("concurrency_limited") for _ in range(4)])
    with pytest.raises(client.PaxaError) as error:
        speak()
    assert error.value.code == "concurrency_limited"
    assert len(fake.calls_to("/v1/tts")) == 4
    assert waits == [2, 4, 8]


def test_lost_connection_retries_with_the_same_key(fake, waits, monkeypatch):
    monkeypatch.setattr(client, "TTS_TIMEOUT", (2.0, 0.3))
    fake.script("POST /v1/tts", Reply(200, b"late", {"content-type": "audio/wav"}, delay=1.0))
    assert speak()[:4] == b"RIFF"
    first, second = fake.calls_to("/v1/tts")
    assert first.headers["idempotency-key"] == second.headers["idempotency-key"]
    assert waits == [1]


def test_body_cut_short_is_retried_with_the_same_key(fake, waits):
    fake.script(
        "POST /v1/tts", Reply(200, wav_bytes(b"\0\0" * 500), {"content-type": "audio/wav"}, cut=50)
    )
    assert len(speak()) > 50
    first, second = fake.calls_to("/v1/tts")
    assert first.headers["idempotency-key"] == second.headers["idempotency-key"]
    assert waits == [1]


def test_odd_code_is_shown_as_sent(fake):
    fake.script(
        "POST /v1/tts",
        Reply(
            400,
            b'{"title":"odd_{0}_code","status":400}',
            {"content-type": "application/problem+json"},
        ),
    )
    with pytest.raises(client.PaxaError, match=r"\(odd_\{0\}_code\)"):
        speak()


def test_unreachable_server(monkeypatch, waits):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("PAXA_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("PAXA_API_KEY", KEY)
    with pytest.raises(client.PaxaError, match="Couldn't reach Paxa") as error:
        speak()
    assert waits == [1, 2, 4]
    assert KEY not in str(error.value)


def test_cancel_in_comfyui_stops_a_wait(monkeypatch):
    class Interrupted(Exception):
        pass

    def throw() -> None:
        raise Interrupted

    comfy = types.ModuleType("comfy")
    management = types.ModuleType("comfy.model_management")
    management.throw_exception_if_processing_interrupted = throw
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    with pytest.raises(Interrupted):
        client._wait(30)


# Speech to text.


def test_transcribe_sends_the_contract_request(fake):
    audio = wav_base64()
    document = transcribe(
        audio_base64=audio,
        language="th",
        subtitles="vtt",
        convention="written",
        vocabulary=["Paxa Labs", "กรุงเทพฯ"],
    )
    (call,) = fake.calls_to("/v1/stt")
    assert call.body == {
        "audio": audio,
        "model": "paxa-stt-lite-v1-preview",
        "subtitles": "vtt",
        "convention": "written",
        "language": "th",
        "vocabulary": ["Paxa Labs", "กรุงเทพฯ"],
    }
    assert call.headers["x-paxa-client"] == client.CLIENT_TAG
    assert document["text"] == fake.transcript
    assert document["subtitles"].startswith("WEBVTT")
    assert document["usage"] == {"seconds": 0.5, "credits": 0.1}


def test_auto_language_and_empty_vocabulary_are_left_out(fake):
    transcribe()
    body = fake.calls_to("/v1/stt")[0].body
    assert "language" not in body
    assert "vocabulary" not in body


def test_slow_transcription_with_keepalive_newlines(fake):
    fake.script(
        "POST /v1/stt",
        Reply(
            200,
            b'{"text":"late but fine","subtitles":"","usage":{"seconds":1,"credits":0.14}}',
            {"content-type": "application/json"},
            drip=3,
        ),
    )
    assert transcribe()["text"] == "late but fine"


def test_slow_transcription_that_fails_inside_200(fake):
    fake.script(
        "POST /v1/stt",
        Reply(
            200,
            b'{"title":"provider_error","status":502}',
            {"content-type": "application/json"},
            drip=2,
        ),
    )
    with pytest.raises(client.PaxaError, match="charge was refunded") as error:
        transcribe()
    assert error.value.code == "provider_error"
    assert error.value.status == 502


def test_unreadable_transcription(fake):
    fake.script("POST /v1/stt", Reply(200, b"{not json", {"content-type": "application/json"}))
    with pytest.raises(client.PaxaError, match="couldn't read"):
        transcribe()


@pytest.mark.parametrize(
    "code", ["audio_invalid", "audio_too_long", "audio_too_large", "content_blocked"]
)
def test_transcription_refusals(fake, code):
    fake.script("POST /v1/stt", problem(code))
    with pytest.raises(client.PaxaError) as error:
        transcribe()
    assert error.value.code == code
    assert error.value.status == STATUS[code]


def test_stand_in_refuses_audio_it_cannot_read(fake):
    with pytest.raises(client.PaxaError) as error:
        transcribe(audio_base64=base64.b64encode(b"not audio").decode())
    assert error.value.code == "audio_invalid"


# Uploads: the connect timeout bounds each block sent, not the whole body.


def largest_recording() -> str:
    """The largest WAV the node sends, 13.65 minutes at 16 kHz, as base64."""
    frames = (STT_MAX_BYTES - 44) // 2
    return base64.b64encode(wav_bytes(b"\0\0" * frames, rate=16_000)).decode()


def test_a_slow_upload_can_outlast_the_connect_timeout(fake, waits, monkeypatch):
    monkeypatch.setattr(client, "STT_TIMEOUT", (0.5, 10.0))
    fake.uplink = Uplink(rate=18_000_000)  # about 2 s for the whole body
    audio = largest_recording()
    started = time.monotonic()
    assert transcribe(audio_base64=audio)["text"] == fake.transcript
    assert time.monotonic() - started > 1.0
    (call,) = fake.calls_to("/v1/stt")
    assert call.body["audio"] == audio
    assert waits == []


def test_a_stalled_upload_still_times_out(fake, waits, monkeypatch):
    monkeypatch.setattr(client, "STT_TIMEOUT", (0.3, 10.0))
    fake.uplink = Uplink(stall_after=1_000_000, stall_for=1.5)
    started = time.monotonic()
    with pytest.raises(client.PaxaError, match="Couldn't reach Paxa"):
        transcribe(audio_base64=largest_recording())
    # Each attempt gives up on its own, before the stand-in would hang up.
    assert time.monotonic() - started < 4 * 1.5
    calls = fake.calls_to("/v1/stt")
    assert len(calls) == 4
    assert len({call.headers["idempotency-key"] for call in calls}) == 1
    assert waits == [1, 2, 4]
