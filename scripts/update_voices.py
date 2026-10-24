"""Refresh comfyui_paxa/voices.json from the public voice list.

GET /v1/voices needs no API key. Run it before a release:

    uv run python scripts/update_voices.py

The nodes read the saved file, so ComfyUI starts without a network call.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comfyui_paxa import client  # noqa: E402

OUT = ROOT / "comfyui_paxa" / "voices.json"


def main() -> None:
    base = os.environ.get(client.BASE_URL_ENV, "").strip().rstrip("/") or client.DEFAULT_BASE_URL
    response = requests.get(
        f"{base}/v1/voices",
        headers={"x-paxa-client": client.CLIENT_TAG, "User-Agent": client.CLIENT_TAG},
        timeout=30,
    )
    response.raise_for_status()
    voices = response.json()["voices"]
    models = {voice["model"] for voice in voices}
    missing = {client.DEFAULT_MODEL, *client.MODELS} - models
    if missing:
        sys.exit(f"The voice list has no voices for {sorted(missing)}; not writing it.")
    snapshot = {
        "source": "GET https://api.paxalabs.com/v1/voices",
        "fetched": datetime.date.today().isoformat(),
        "voices": voices,
    }
    OUT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = ", ".join(
        f"{model}: {sum(v['model'] == model for v in voices)}" for model in sorted(models)
    )
    print(f"Wrote {OUT.relative_to(ROOT)} ({counts}).")


if __name__ == "__main__":
    main()
