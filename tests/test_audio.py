from __future__ import annotations

import io
import struct
import sys
import types
import wave

import numpy as np
import pytest
import torch
from fake_paxa import tone, wav_bytes

from comfyui_paxa import audio

LIMITS = {"max_bytes": 26_214_400, "max_seconds": 3_600}


def pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def read_wav(data: bytes) -> tuple[int, int, np.ndarray]:
    with wave.open(io.BytesIO(data)) as file:
        frames = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
        return file.getnchannels(), file.getframerate(), frames


# WAV from Paxa to AUDIO.


def test_paxa_wav_becomes_audio():
    result = audio.wav_to_audio(wav_bytes(pcm(0, 16384, -32768, 32767)))
    assert result["sample_rate"] == 24_000
    waveform = result["waveform"]
    assert waveform.dtype == torch.float32
    assert waveform.shape == (1, 1, 4)
    assert waveform[0, 0].tolist() == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])


def test_streamed_wav_sizes_read_to_the_end():
    data = wav_bytes(pcm(1, 2, 3), streamed=True) + b"\x07"  # a stray half sample
    assert audio.wav_to_audio(data)["waveform"].shape == (1, 1, 3)


def test_stereo_is_deinterleaved():
    data = wav_bytes(pcm(100, -100, 200, -200), rate=48_000, channels=2)
    waveform = audio.wav_to_audio(data)["waveform"]
    assert waveform.shape == (1, 2, 2)
    assert (waveform[0, 0] * 32768).tolist() == [100, 200]
    assert (waveform[0, 1] * 32768).tolist() == [-100, -200]


def test_other_chunks_before_data_are_skipped():
    plain = wav_bytes(pcm(5, 6))
    extra = b"LIST" + struct.pack("<I", 3) + b"abc\x00"  # odd size, padded
    data = plain[:36] + extra + plain[36:]
    assert (audio.wav_to_audio(data)["waveform"][0, 0] * 32768).tolist() == [5, 6]


def test_extensible_pcm_is_read():
    fmt = struct.pack("<HHIIHH", 0xFFFE, 1, 24_000, 48_000, 2, 16)
    fmt += struct.pack("<HHI", 22, 16, 4) + struct.pack("<H", 1) + b"\x00" * 14
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", 4)
        + pcm(7, 8)
    )
    data = b"RIFF" + struct.pack("<I", len(body)) + body
    assert (audio.wav_to_audio(data)["waveform"][0, 0] * 32768).tolist() == [7, 8]


@pytest.mark.parametrize(
    "data, message",
    [
        (b"ID3 not a wav file at all", "isn't a WAV"),
        (b"RIFF" + struct.pack("<I", 4) + b"WAVE", "no audio data"),
        (b"RIFF\x00\x00\x00\x00WAVEdata\x02\x00\x00\x00\x01\x02", "data before its format"),
        (wav_bytes(b""), "holds no audio"),
    ],
)
def test_broken_wav_is_refused(data, message):
    with pytest.raises(ValueError, match=message):
        audio.wav_to_audio(data)


def test_only_16_bit_pcm_is_read():
    data = bytearray(wav_bytes(pcm(1, 2)))
    struct.pack_into("<H", data, 34, 8)  # bits per sample
    with pytest.raises(ValueError, match="16-bit PCM"):
        audio.wav_to_audio(bytes(data))


# AUDIO to WAV for transcription.


def test_audio_becomes_mono_16_bit_wav():
    left = torch.tensor([0.5, -0.5, 1.0, 0.25])
    right = torch.tensor([0.5, 0.5, 1.0, -0.25])
    waveform = torch.stack([left, right]).unsqueeze(0)
    data, seconds = audio.audio_to_wav({"waveform": waveform, "sample_rate": 8}, **LIMITS)
    channels, rate, frames = read_wav(data)
    assert (channels, rate) == (1, 8)
    assert frames.tolist() == [16384, 0, 32767, 0]
    assert seconds == 0.5


def test_round_trip_keeps_the_signal():
    original = audio.wav_to_audio(wav_bytes(tone(0.1)))
    data, _ = audio.audio_to_wav(original, **LIMITS)
    again = audio.wav_to_audio(data)
    assert again["sample_rate"] == 24_000
    assert torch.allclose(again["waveform"], original["waveform"], atol=1 / 16384)


@pytest.mark.parametrize("shape", [(4,), (1, 4)])
def test_unbatched_waveforms_are_accepted(shape):
    data, _ = audio.audio_to_wav({"waveform": torch.zeros(shape), "sample_rate": 4}, **LIMITS)
    assert read_wav(data)[2].size == 4


def test_an_unexpected_shape_is_refused():
    with pytest.raises(ValueError, match=r"shape \(1, 1, 1, 4\)"):
        audio.audio_to_wav({"waveform": torch.zeros(1, 1, 1, 4), "sample_rate": 4}, **LIMITS)


def test_a_batch_is_refused():
    with pytest.raises(ValueError, match="batch of 2"):
        audio.audio_to_wav({"waveform": torch.zeros(2, 1, 4), "sample_rate": 4}, **LIMITS)


def test_empty_audio_is_refused():
    with pytest.raises(ValueError, match="empty"):
        audio.audio_to_wav({"waveform": torch.zeros(1, 1, 0), "sample_rate": 16_000}, **LIMITS)


def test_longer_than_an_hour_is_refused():
    with pytest.raises(ValueError, match="up to 60 minutes"):
        audio.audio_to_wav({"waveform": torch.zeros(1, 1, 3_601), "sample_rate": 1}, **LIMITS)


def fake_comfy_resample(monkeypatch, calls):
    def resample(waveform, source, target):
        calls.append((source, target))
        return waveform[..., :: source // target]

    comfy = types.ModuleType("comfy")
    comfy_audio = types.ModuleType("comfy.audio")
    comfy_audio.resample = resample
    comfy.audio = comfy_audio
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.audio", comfy_audio)


def test_too_large_at_its_own_rate_is_resampled_to_16_khz(monkeypatch):
    calls = []
    fake_comfy_resample(monkeypatch, calls)
    clip = {"waveform": torch.zeros(1, 1, 48_000), "sample_rate": 48_000}
    data, seconds = audio.audio_to_wav(clip, max_bytes=60_000, max_seconds=3_600)
    assert calls == [(48_000, 16_000)]
    channels, rate, frames = read_wav(data)
    assert (channels, rate, frames.size) == (1, 16_000, 16_000)
    assert seconds == 1.0


def test_small_audio_keeps_its_rate(monkeypatch):
    calls = []
    fake_comfy_resample(monkeypatch, calls)
    clip = {"waveform": torch.zeros(1, 1, 4_800), "sample_rate": 48_000}
    _, rate, _ = read_wav(audio.audio_to_wav(clip, **LIMITS)[0])
    assert rate == 48_000
    assert calls == []


def test_too_large_even_at_16_khz_is_refused(monkeypatch):
    fake_comfy_resample(monkeypatch, [])
    clip = {"waveform": torch.zeros(1, 1, 48_000), "sample_rate": 48_000}
    with pytest.raises(ValueError, match="Trim it"):
        audio.audio_to_wav(clip, max_bytes=20_000, max_seconds=3_600)


def test_without_a_resampler_the_message_says_trim(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.audio", None)
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    monkeypatch.setitem(sys.modules, "torchaudio.functional", None)
    clip = {"waveform": torch.zeros(1, 1, 48_000), "sample_rate": 48_000}
    with pytest.raises(ValueError, match="no resampler"):
        audio.audio_to_wav(clip, max_bytes=60_000, max_seconds=3_600)
