from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_paxa import KEY, FakePaxa  # noqa: E402

from comfyui_paxa import client  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def voices() -> list[dict]:
    return json.loads((ROOT / "comfyui_paxa" / "voices.json").read_text(encoding="utf-8"))["voices"]


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test reads a real key, talks to the real API, or keeps a session."""
    monkeypatch.delenv(client.KEY_ENV, raising=False)
    monkeypatch.delenv(client.BASE_URL_ENV, raising=False)
    monkeypatch.setattr(client, "KEY_FILE", tmp_path / "paxa_api_key.txt")
    monkeypatch.setattr(client, "_session", None)


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Records retry waits instead of sleeping."""
    seen: list[float] = []
    monkeypatch.setattr(client, "_wait", seen.append)
    return seen


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, voices: list[dict]):
    server = FakePaxa(voices).start()
    monkeypatch.setenv(client.BASE_URL_ENV, server.url)
    monkeypatch.setenv(client.KEY_ENV, KEY)
    yield server
    server.stop()
