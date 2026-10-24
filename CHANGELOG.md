# Changelog

## 0.1.0

First release.

- **Paxa Text to Speech:** text to `AUDIO` with Paxa TTS Flash v2 or v1, the
  bundled voice list or any voice id, speed, and language.
- **Paxa Speech to Text (preview):** `AUDIO` to a transcript and SRT or
  WebVTT subtitles with Paxa STT Lite, a research preview.
- **Paxa Save Subtitles:** writes `.srt` or `.vtt` files to ComfyUI's output
  folder.
- The API key comes from `PAXA_API_KEY` or `paxa_api_key.txt`, never from a
  node input.
- Rate limits, concurrency limits and a briefly unavailable model are retried
  with the same idempotency key.
- Two example workflows for ComfyUI's template browser.
