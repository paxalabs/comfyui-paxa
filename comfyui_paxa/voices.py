"""The voice list the TTS node offers, read from a snapshot of GET /v1/voices.

The snapshot ships with the nodes, so ComfyUI starts without a network
call. scripts/update_voices.py refreshes it. Voices newer than the
snapshot still work through the node's voice_id input.
"""

from __future__ import annotations

import json
from pathlib import Path

from .client import DEFAULT_MODEL

SNAPSHOT = Path(__file__).resolve().parent / "voices.json"
# The two voices Paxa recommends as defaults, female then male.
RECOMMENDED = ("foithong", "massaman")


def load() -> list[dict]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))["voices"]


def voice_ids(voices: list[dict]) -> list[str]:
    """Every voice id once: the recommended pair, the rest of the default
    model's roster, then the voices only the older model speaks."""
    default = [voice["id"] for voice in voices if voice["model"] == DEFAULT_MODEL]
    ordered = [*RECOMMENDED, *(voice for voice in default if voice not in RECOMMENDED)]
    seen = set(ordered)
    for voice in voices:
        if voice["id"] not in seen:
            ordered.append(voice["id"])
            seen.add(voice["id"])
    return ordered


def models_by_voice(voices: list[dict]) -> dict[str, set[str]]:
    served: dict[str, set[str]] = {}
    for voice in voices:
        served.setdefault(voice["id"], set()).add(voice["model"])
    return served
