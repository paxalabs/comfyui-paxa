"""The Paxa nodes: text to speech, speech to text, and saving subtitles."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from . import client, voices
from .audio import audio_to_wav, wav_to_audio

log = logging.getLogger("comfyui-paxa")

CATEGORY = "Paxa"
LANGUAGES = ["auto", "th", "en", "zh"]
DEFAULT_TEXT = "สวัสดีค่ะ ยินดีต้อนรับสู่ Paxa Labs. Hello from ComfyUI! 你好！"
VOICE_ID_MAX = 100
# paxa-stt-lite-v1-preview's ceilings, as GET /v1/models reports them.
STT_MAX_BYTES = 26_214_400
STT_MAX_SECONDS = 3_600
VOCABULARY_MAX_TERMS = 50
VOCABULARY_MAX_CHARS = 50

_VOICES = voices.load()
VOICE_IDS = voices.voice_ids(_VOICES)
_MODELS_BY_VOICE = voices.models_by_voice(_VOICES)
V1_ONLY = [voice for voice in VOICE_IDS if client.DEFAULT_MODEL not in _MODELS_BY_VOICE[voice]]


class PaxaTextToSpeech:
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Speaks text in Thai, English or Mandarin Chinese, or a mix of them, in a Paxa voice. "
        "Calls the Paxa API with your key (PAXA_API_KEY, or paxa_api_key.txt in this node "
        "pack's folder) and spends your Paxa credits."
    )
    FUNCTION = "speak"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_TOOLTIPS = ("Mono speech, as the 24 kHz WAV Paxa sends.",)
    SEARCH_ALIASES = ["tts", "text to speech", "voiceover", "voice", "thai", "mandarin", "paxa"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": DEFAULT_TEXT,
                        "tooltip": (
                            "Thai, English, Mandarin Chinese or any mix, up to 5,000 characters. "
                            "On paxa-tts-flash-v2, direction tags such as [whispering] or "
                            "[laughs] set the delivery."
                        ),
                    },
                ),
                "voice": (
                    VOICE_IDS,
                    {
                        "default": voices.RECOMMENDED[0],
                        "tooltip": (
                            "Every paxa-tts-flash-v2 voice speaks Thai, English and Mandarin. "
                            f"These speak on paxa-tts-flash-v1 only: {', '.join(V1_ONLY)}. "
                            "Hear them at paxalabs.com/voices."
                        ),
                    },
                ),
                "model": (
                    list(client.MODELS),
                    {
                        "default": client.DEFAULT_MODEL,
                        "tooltip": (
                            "paxa-tts-flash-v2 is the current model. paxa-tts-flash-v1 has the "
                            "regional Thai accents."
                        ),
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 1.5,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": (
                            "Speaking rate; 1.0 is the voice as designed. "
                            "Costs the same at any rate."
                        ),
                    },
                ),
                "language": (
                    LANGUAGES,
                    {
                        "default": "auto",
                        "tooltip": (
                            "How numbers, codes and symbols are read. Words always keep the "
                            "language they're written in."
                        ),
                    },
                ),
            },
            "optional": {
                "voice_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Leave empty to use the voice list. Or type any voice id from "
                            "paxalabs.com/voices, including your own designed voice (voice_...)."
                        ),
                    },
                ),
            },
        }

    def speak(
        self,
        text: str,
        voice: str,
        model: str,
        speed: float,
        language: str,
        voice_id: str = "",
    ) -> tuple[dict[str, Any]]:
        if not text.strip():
            raise ValueError("Paxa Text to Speech: the text is empty.")
        chosen = voice_id.strip() or voice
        if len(chosen) > VOICE_ID_MAX:
            raise ValueError(
                f"Paxa Text to Speech: a voice id has at most {VOICE_ID_MAX} characters."
            )
        wav = client.synthesize(
            text=text,
            voice=chosen,
            model=model,
            speed=speed,
            language=language,
            voice_hint=_voice_hint(chosen, model),
        )
        return (wav_to_audio(wav),)


class PaxaSpeechToText:
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Research preview. Transcribes speech, Thai first, and writes SRT or WebVTT subtitles "
        "that break at word boundaries, Thai included. Calls the Paxa API with your key and "
        "spends your Paxa credits."
    )
    EXPERIMENTAL = True
    FUNCTION = "transcribe"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "subtitles")
    OUTPUT_TOOLTIPS = ("The transcript.", "The subtitle file's text, SRT or WebVTT.")
    SEARCH_ALIASES = ["stt", "speech to text", "transcribe", "subtitles", "captions", "srt", "paxa"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "One recording, up to about 13 minutes."}),
                "language": (
                    LANGUAGES,
                    {
                        "default": "auto",
                        "tooltip": (
                            "The language you expect. It's a hint, not a filter, and auto "
                            "detects it."
                        ),
                    },
                ),
                "subtitle_format": (["srt", "vtt"], {"default": "srt"}),
                "convention": (
                    ["spoken", "written"],
                    {
                        "default": "spoken",
                        "tooltip": (
                            "spoken writes numbers as they were said. written uses digits and "
                            "symbols, as in 30%."
                        ),
                    },
                ),
            },
            "optional": {
                "vocabulary": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": (
                            "Names and terms the speech is likely to contain, one per line, "
                            f"up to {VOCABULARY_MAX_TERMS}."
                        ),
                    },
                ),
            },
        }

    def transcribe(
        self,
        audio: dict[str, Any],
        language: str,
        subtitle_format: str,
        convention: str,
        vocabulary: str = "",
    ) -> tuple[str, str]:
        terms = _vocabulary(vocabulary)
        wav, _ = audio_to_wav(audio, max_bytes=STT_MAX_BYTES, max_seconds=STT_MAX_SECONDS)
        document = client.transcribe(
            audio_base64=base64.b64encode(wav).decode("ascii"),
            language=language,
            subtitles=subtitle_format,
            convention=convention,
            vocabulary=terms,
        )
        return (document["text"], document.get("subtitles") or "")


class PaxaSaveSubtitles:
    CATEGORY = CATEGORY
    DESCRIPTION = "Saves subtitles as an .srt or .vtt file in ComfyUI's output folder."
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    SEARCH_ALIASES = ["save subtitles", "save srt", "srt", "vtt", "captions"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "subtitles": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "paxa/subtitles"}),
            },
        }

    def save(self, subtitles: str, filename_prefix: str) -> dict[str, Any]:
        import folder_paths

        extension = "vtt" if subtitles.lstrip("﻿").startswith("WEBVTT") else "srt"
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        file = f"{name}_{counter:05}_.{extension}"
        with open(os.path.join(folder, file), "w", encoding="utf-8", newline="") as handle:
            handle.write(subtitles)
        log.info("Paxa: saved subtitles to %s", os.path.join(subfolder, file))
        return {
            "ui": {
                "text": (subtitles,),
                "files": [{"filename": file, "subfolder": subfolder, "type": "output"}],
            }
        }


def _voice_hint(voice: str, model: str) -> str:
    served = _MODELS_BY_VOICE.get(voice.lower())
    if served and model not in served:
        return f"It speaks on {', '.join(sorted(served))}; switch the model. "
    if not served:
        return "Check the voice id. "
    return ""


def _vocabulary(text: str) -> list[str]:
    terms = list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))
    if len(terms) > VOCABULARY_MAX_TERMS:
        raise ValueError(
            f"Paxa Speech to Text: the vocabulary has {len(terms)} terms; the limit is "
            f"{VOCABULARY_MAX_TERMS}."
        )
    long_terms = [term for term in terms if len(term) > VOCABULARY_MAX_CHARS]
    if long_terms:
        raise ValueError(
            f"Paxa Speech to Text: vocabulary terms have at most {VOCABULARY_MAX_CHARS} "
            f"characters; this one is longer: {long_terms[0][:60]}"
        )
    return terms


NODE_CLASS_MAPPINGS = {
    "Paxa_TextToSpeech": PaxaTextToSpeech,
    "Paxa_SpeechToText": PaxaSpeechToText,
    "Paxa_SaveSubtitles": PaxaSaveSubtitles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Paxa_TextToSpeech": "Paxa Text to Speech",
    "Paxa_SpeechToText": "Paxa Speech to Text (preview)",
    "Paxa_SaveSubtitles": "Paxa Save Subtitles",
}
