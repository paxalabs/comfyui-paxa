# Paxa for ComfyUI

Text to speech in Thai, English and Mandarin Chinese, plus speech to text
with subtitles, from [Paxa Labs](https://paxalabs.com) as ComfyUI nodes.

| Node | In | Out |
|---|---|---|
| **Paxa Text to Speech** | text | `AUDIO` |
| **Paxa Speech to Text (preview)** | `AUDIO` | transcript, SRT or WebVTT subtitles |
| **Paxa Save Subtitles** | subtitles | an `.srt` or `.vtt` file in ComfyUI's output folder |

- **50 voices** on Paxa TTS Flash v2, and every one speaks Thai, English and
  Mandarin Chinese, including sentences that mix them. Hear them at
  [paxalabs.com/voices](https://paxalabs.com/voices).
- **Direction tags** in the text set the delivery, for example
  `[whispering]` or `[warm, unhurried]`. The tag guide is at
  [paxalabs.com/docs/directions](https://paxalabs.com/docs/directions).
- **Subtitles that break at word boundaries**, Thai included, so a cue
  never splits a Thai word.

The nodes call the Paxa API with your own Paxa API key and spend credits
from your Paxa account. New accounts include free credits.

## Install

- **ComfyUI Manager:** search for "Paxa", install, and restart ComfyUI.
- **Comfy CLI:** `comfy node install comfyui-paxa`
- **By hand:** clone this repository into `ComfyUI/custom_nodes`, install
  `requirements.txt` with the Python that runs ComfyUI, and restart.

The only extra dependency is `requests`. The nodes use ComfyUI's own torch
and numpy.

## Your API key

1. Create a key at [paxalabs.com/app/keys](https://paxalabs.com/app/keys).
2. Give it to ComfyUI in one of two ways:
   - **Environment variable:** set `PAXA_API_KEY` before you start ComfyUI.

     ```bash
     export PAXA_API_KEY=pxa_...        # macOS and Linux
     $env:PAXA_API_KEY = "pxa_..."      # Windows PowerShell
     ```

   - **Key file:** save the key, alone, in a file named `paxa_api_key.txt`
     in this node pack's folder, `ComfyUI/custom_nodes/comfyui-paxa/`. This
     suits the portable and desktop builds. Updates through ComfyUI Manager
     keep the file. On Windows, check that it isn't saved as
     `paxa_api_key.txt.txt`.

   The environment variable wins when both are set. The nodes read the key
   each time they run, so a new key file works without a restart.

**Why there's no key field on the nodes:** ComfyUI saves every node input in
the workflow, and in the metadata of every image and audio file it saves. A
key typed into a node would travel with every workflow and output you share.

## Nodes

### Paxa Text to Speech

| Input | What it does |
|---|---|
| `text` | Thai, English, Mandarin Chinese or any mix, up to 5,000 characters per run. |
| `voice` | The voice. `foithong` (female) and `massaman` (male) come first. The last eight in the list speak on Paxa TTS Flash (v1) only: `padthai`, `yoyo`, `somtam`, `larb`, `khaosoi`, `roti`, `donut` and `toast`. |
| `model` | `paxa-tts-flash-v2`, the current model, or `paxa-tts-flash-v1`, which has the regional Thai accents. |
| `speed` | 0.5 to 1.5; 1.0 is the voice as designed. Every speed costs the same. |
| `language` | How numbers, codes and symbols are read: `auto`, `th`, `en` or `zh`. Words always keep the language they're written in. |
| `voice_id` | Optional. Any voice id, including one newer than this pack or your own designed voice (`voice_...`). It replaces `voice` when set. |

The output is mono speech, the 24 kHz WAV Paxa sends. ComfyUI reuses a
node's last result while its inputs stay the same, so running the workflow
again doesn't call Paxa or spend credits until you change an input.

### Paxa Speech to Text (preview)

It runs Paxa STT Lite, a research preview: Thai-first transcription that
also handles English, Mandarin and speech that switches language mid
sentence. Its behavior may change while it's in preview.

| Input | What it does |
|---|---|
| `audio` | One recording. Stereo is mixed down to mono. |
| `language` | The language you expect: a hint, not a filter. `auto` detects it. |
| `subtitle_format` | `srt` or `vtt`. |
| `convention` | `spoken` writes numbers as they were said; `written` uses digits and symbols, as in 30%. |
| `vocabulary` | Optional names and terms the speech is likely to contain, one per line, up to 50. |

The outputs are the transcript and the subtitle file's text. The node sends
up to about 13 minutes per run; trim longer recordings with ComfyUI's Trim
Audio Duration node.

### Paxa Save Subtitles

Saves the subtitles under `ComfyUI/output/`, as `.vtt` for WebVTT and
`.srt` otherwise. `filename_prefix` works like the one on Save Image.

## Example workflows

In ComfyUI, open **Browse Templates** and find **comfyui-paxa**:

- **Paxa Text to Speech** speaks a line in Thai, English and Mandarin and
  saves it as MP3.
- **Paxa Voiceover with Subtitles** speaks a Thai script, saves it, and
  turns it into SRT subtitles.

## When something goes wrong

The node's error message says what Paxa answered and how to fix it, with the
request id to quote if you contact Paxa. For example:

- **No Paxa API key found:** set the key as described above.
- **Paxa didn't accept the API key:** check the key at
  [paxalabs.com/app/keys](https://paxalabs.com/app/keys).
- **The voice isn't available on this model:** pick a voice the model
  speaks, or switch the model.
- **Not enough credits:** nothing was charged; top up and run again.

When your plan's rate limit or concurrency limit is reached, or the model is
briefly unavailable, the node waits and retries on its own. Every retry
carries the same idempotency key, so a retry is never charged twice.

## Connection and data

The nodes connect to one host, `api.paxalabs.com`, over HTTPS. Your text
and audio go to Paxa to be processed, under Paxa's
[privacy policy](https://paxalabs.com/privacy). Each request names this pack
and its version in an `x-paxa-client` header, so Paxa can count how much use
comes through ComfyUI.

`PAXA_BASE_URL` points the nodes at another Paxa endpoint, such as a
dedicated deployment. It must be an `https://` address.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The tests run against a local stand-in for the Paxa API and need no key.
`uv run python scripts/update_voices.py` refreshes the bundled voice list
from the public `GET /v1/voices`.

## License

MIT. See [LICENSE](LICENSE).
