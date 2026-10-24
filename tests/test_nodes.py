from __future__ import annotations

import base64
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from fake_paxa import problem, tone

from comfyui_paxa import client, nodes
from comfyui_paxa.nodes import PaxaSaveSubtitles, PaxaSpeechToText, PaxaTextToSpeech

ROOT = Path(__file__).resolve().parent.parent
WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}


def widgets(node_class) -> list[tuple[str, object]]:
    """The inputs ComfyUI draws as widgets, in order."""
    found = []
    for section in ("required", "optional"):
        for name, spec in node_class.INPUT_TYPES().get(section, {}).items():
            kind = spec[0]
            options = spec[1] if len(spec) > 1 else {}
            if isinstance(kind, list) or (kind in WIDGET_TYPES and not options.get("forceInput")):
                found.append((name, spec))
    return found


def load_pack_like_comfyui():
    """Imports the pack folder the way ComfyUI does: by path, as a package."""
    spec = importlib.util.spec_from_file_location(
        "paxa_pack_under_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pack_exports_the_three_nodes():
    pack = load_pack_like_comfyui()
    assert set(pack.NODE_CLASS_MAPPINGS) == {
        "Paxa_TextToSpeech",
        "Paxa_SpeechToText",
        "Paxa_SaveSubtitles",
    }
    assert pack.NODE_DISPLAY_NAME_MAPPINGS == {
        "Paxa_TextToSpeech": "Paxa Text to Speech",
        "Paxa_SpeechToText": "Paxa Speech to Text (preview)",
        "Paxa_SaveSubtitles": "Paxa Save Subtitles",
    }


@pytest.mark.parametrize("node_class", list(nodes.NODE_CLASS_MAPPINGS.values()))
def test_node_definitions(node_class):
    assert node_class.CATEGORY == "Paxa"
    assert callable(getattr(node_class(), node_class.FUNCTION))
    assert isinstance(node_class.RETURN_TYPES, tuple)
    # Partner nodes are Comfy's own, billed in Comfy credits. These aren't.
    assert not hasattr(node_class, "API_NODE")
    for name, spec in widgets(node_class):
        if isinstance(spec[0], list):
            assert spec[1]["default"] in spec[0], name


def test_preview_is_flagged_experimental():
    assert PaxaSpeechToText.EXPERIMENTAL is True
    assert "preview" in nodes.NODE_DISPLAY_NAME_MAPPINGS["Paxa_SpeechToText"]
    assert not getattr(PaxaTextToSpeech, "EXPERIMENTAL", False)


def test_text_inputs_skip_dynamic_prompts():
    # The frontend would otherwise turn {a|b} in the text into a random pick.
    assert PaxaTextToSpeech.INPUT_TYPES()["required"]["text"][1]["dynamicPrompts"] is False
    assert PaxaSpeechToText.INPUT_TYPES()["optional"]["vocabulary"][1]["dynamicPrompts"] is False


def test_no_input_takes_an_api_key():
    for node_class in nodes.NODE_CLASS_MAPPINGS.values():
        for section in node_class.INPUT_TYPES().values():
            assert not any("key" in name.lower() for name in section)


def test_voice_list(voices):
    ids = nodes.VOICE_IDS
    assert ids[:2] == ["foithong", "massaman"]
    assert len(ids) == len(set(ids)) == len({voice["id"] for voice in voices})
    v2 = {voice["id"] for voice in voices if voice["model"] == "paxa-tts-flash-v2"}
    assert set(ids[: len(v2)]) == v2
    assert set(nodes.V1_ONLY) == set(ids[len(v2) :])
    assert nodes.V1_ONLY and all(voice not in v2 for voice in nodes.V1_ONLY)
    assert PaxaTextToSpeech.INPUT_TYPES()["required"]["voice"][1]["default"] == "foithong"


# Text to speech.


def test_text_to_speech_returns_comfyui_audio(fake):
    (result,) = PaxaTextToSpeech().speak("Hello", "foithong", "paxa-tts-flash-v2", 1.0, "auto")
    assert result["sample_rate"] == 24_000
    assert result["waveform"].dtype == torch.float32
    assert result["waveform"].dim() == 3 and result["waveform"].shape[:2] == (1, 1)
    assert fake.calls_to("/v1/tts")[0].body["voice"] == "foithong"


def test_voice_id_overrides_the_list(fake):
    PaxaTextToSpeech().speak(
        "Hi", "foithong", "paxa-tts-flash-v2", 1.0, "auto", voice_id=" massaman "
    )
    assert fake.calls_to("/v1/tts")[0].body["voice"] == "massaman"


def test_v1_voice_on_v2_explains_the_fix(fake):
    with pytest.raises(client.PaxaError, match="It speaks on paxa-tts-flash-v1; switch the model"):
        PaxaTextToSpeech().speak("Hi", "somtam", "paxa-tts-flash-v2", 1.0, "auto")


def test_unknown_voice_id_asks_to_check_it(fake):
    with pytest.raises(client.PaxaError, match="Check the voice id"):
        PaxaTextToSpeech().speak("Hi", "foithong", "paxa-tts-flash-v2", 1.0, "auto", "nosuchvoice")


def test_v1_voice_on_v1(fake):
    PaxaTextToSpeech().speak("สวัสดีเจ้า", "somtam", "paxa-tts-flash-v1", 1.0, "th")
    assert fake.calls_to("/v1/tts")[0].body["model"] == "paxa-tts-flash-v1"


@pytest.mark.parametrize(
    "text, voice_id, message",
    [("  \n ", "", "text is empty"), ("Hi", "v" * 101, "at most 100 characters")],
)
def test_text_to_speech_checks_before_calling(fake, text, voice_id, message):
    with pytest.raises(ValueError, match=message):
        PaxaTextToSpeech().speak(text, "foithong", "paxa-tts-flash-v2", 1.0, "auto", voice_id)
    assert fake.calls == []


# Speech to text.


def clip(seconds: float = 0.5) -> dict:
    samples = np.frombuffer(tone(seconds), dtype="<i2").astype(np.float32) / 32768
    return {"waveform": torch.from_numpy(samples).reshape(1, 1, -1), "sample_rate": 24_000}


def test_speech_to_text_returns_text_and_subtitles(fake):
    text, subtitles = PaxaSpeechToText().transcribe(clip(), "th", "srt", "written", "")
    assert text == fake.transcript
    assert subtitles.startswith("1\n00:00:00,000 --> ")
    body = fake.calls_to("/v1/stt")[0].body
    assert body["language"] == "th"
    assert body["convention"] == "written"
    sent = base64.b64decode(body["audio"])
    assert sent[:4] == b"RIFF" and sent[8:12] == b"WAVE"


def test_vocabulary_is_one_term_per_line(fake):
    PaxaSpeechToText().transcribe(
        clip(), "auto", "vtt", "spoken", "Paxa Labs\n\n  ฟอยทอง \nPaxa Labs\n"
    )
    assert fake.calls_to("/v1/stt")[0].body["vocabulary"] == ["Paxa Labs", "ฟอยทอง"]


@pytest.mark.parametrize(
    "vocabulary, message",
    [
        ("\n".join(f"term {n}" for n in range(51)), "has 51 terms"),
        ("x" * 51, "at most 50 characters"),
    ],
)
def test_vocabulary_limits_are_checked_before_calling(fake, vocabulary, message):
    with pytest.raises(ValueError, match=message):
        PaxaSpeechToText().transcribe(clip(), "auto", "srt", "spoken", vocabulary)
    assert fake.calls == []


def test_transcription_refusal_reaches_the_user(fake):
    fake.script("POST /v1/stt", problem("insufficient_credits"))
    with pytest.raises(client.PaxaError, match="enough credits"):
        PaxaSpeechToText().transcribe(clip(), "auto", "srt", "spoken", "")


def test_round_trip_through_the_paxa_wav(fake):
    (spoken,) = PaxaTextToSpeech().speak("Hello there", "foithong", "paxa-tts-flash-v2", 1.0, "en")
    text, _ = PaxaSpeechToText().transcribe(spoken, "auto", "srt", "spoken", "")
    sent = base64.b64decode(fake.calls_to("/v1/stt")[0].body["audio"])
    assert sent[24:28] == (24_000).to_bytes(4, "little")
    assert text == fake.transcript


# Saving subtitles.


@pytest.fixture
def output_dir(monkeypatch, tmp_path):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_output_directory = lambda: str(tmp_path)

    def get_save_image_path(prefix, output_dir):
        subfolder, name = Path(prefix).parent, Path(prefix).name
        full = Path(output_dir) / subfolder
        full.mkdir(parents=True, exist_ok=True)
        counter = len(list(full.glob(f"{name}_*_.*"))) + 1
        return str(full), name, counter, str(subfolder), prefix

    folder_paths.get_save_image_path = get_save_image_path
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    return tmp_path


def test_save_subtitles_writes_srt_and_vtt(output_dir):
    srt = "1\r\n00:00:00,000 --> 00:00:01,000\r\nสวัสดี\r\n"
    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nสวัสดี\n"
    first = PaxaSaveSubtitles().save(srt, "paxa/subtitles")
    second = PaxaSaveSubtitles().save(vtt, "paxa/subtitles")
    assert first["ui"]["files"] == [
        {"filename": "subtitles_00001_.srt", "subfolder": "paxa", "type": "output"}
    ]
    assert second["ui"]["files"][0]["filename"] == "subtitles_00002_.vtt"
    assert (output_dir / "paxa" / "subtitles_00001_.srt").read_bytes() == srt.encode()
    assert (output_dir / "paxa" / "subtitles_00002_.vtt").read_text(encoding="utf-8") == vtt
