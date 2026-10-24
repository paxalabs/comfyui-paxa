"""Converts between WAV bytes and ComfyUI's AUDIO value.

AUDIO is a dict: "waveform" is a float tensor shaped [batch, channels,
samples] in -1..1, and "sample_rate" is an int.
"""

from __future__ import annotations

import io
import struct
import wave
from typing import Any

import numpy as np
import torch

WAVE_FORMAT_PCM = 1
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
# A streamed WAV can't know its length up front, so it says 0xFFFFFFFF.
UNKNOWN_SIZE = 0xFFFFFFFF
STT_RATE = 16_000


def wav_to_audio(data: bytes) -> dict[str, Any]:
    """Reads 16-bit PCM WAV bytes into an AUDIO value with a batch of one."""
    channels, rate, pcm = _read_pcm16(data)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    frames = samples.reshape(-1, channels).T.copy()
    return {"waveform": torch.from_numpy(frames).unsqueeze(0), "sample_rate": rate}


def audio_to_wav(
    audio: dict[str, Any], *, max_bytes: int, max_seconds: float
) -> tuple[bytes, float]:
    """Writes one recording as 16-bit mono WAV for transcription.

    Returns the file and its length in seconds. Stereo is mixed down. A
    recording too large for one request at its own rate is resampled to
    16 kHz, the rate speech recognition reads anyway.
    """
    waveform = audio["waveform"]
    rate = int(audio["sample_rate"])
    if waveform.dim() == 3:
        if waveform.shape[0] != 1:
            raise ValueError(
                f"Paxa transcribes one recording at a time, and this AUDIO holds a batch of "
                f"{waveform.shape[0]}. Split the batch first."
            )
        waveform = waveform[0]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 2:
        raise ValueError(
            f"The AUDIO waveform has shape {tuple(waveform.shape)}; ComfyUI audio is "
            "[batch, channels, samples]."
        )
    mono = waveform.detach().to(device="cpu", dtype=torch.float32).mean(dim=0, keepdim=True)
    frames = mono.shape[-1]
    if frames == 0 or rate <= 0:
        raise ValueError("The AUDIO input is empty.")
    seconds = frames / rate
    if seconds > max_seconds:
        raise ValueError(
            f"The recording is {seconds / 60:.1f} minutes long, and Paxa transcribes up to "
            f"{max_seconds / 60:.0f} minutes per request. Trim it with Trim Audio Duration."
        )
    if _wav_size(frames) > max_bytes and rate > STT_RATE:
        mono = _resample(mono, rate, STT_RATE)
        rate = STT_RATE
        frames = mono.shape[-1]
    if _wav_size(frames) > max_bytes:
        limit = (max_bytes - 44) / (2 * STT_RATE) / 60
        raise ValueError(
            f"The recording is {seconds / 60:.1f} minutes long. This node sends up to "
            f"{limit:.0f} minutes per request. Trim it with Trim Audio Duration."
        )
    pcm = (mono[0].clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).numpy()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(pcm.astype("<i2").tobytes())
    return buffer.getvalue(), seconds


def _wav_size(frames: int) -> int:
    return 44 + 2 * frames


def _read_pcm16(data: bytes) -> tuple[int, int, bytes]:
    """Returns (channels, sample rate, PCM bytes) from a 16-bit PCM WAV."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Paxa's audio isn't a WAV file.")
    fmt: tuple[int, ...] | None = None
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        size = struct.unpack_from("<I", data, position + 4)[0]
        start = position + 8
        if chunk_id == b"fmt " and start + 16 <= len(data):
            fmt = struct.unpack_from("<HHIIHH", data, start)
            if fmt[0] == WAVE_FORMAT_EXTENSIBLE and size >= 40:
                # The subformat GUID starts with the real format code.
                fmt = (struct.unpack_from("<H", data, start + 24)[0], *fmt[1:])
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("Paxa's WAV has its data before its format.")
            audio_format, channels, rate, _, _, bits = fmt
            if audio_format != WAVE_FORMAT_PCM or bits != 16 or channels < 1:
                raise ValueError(
                    f"Paxa's WAV is format {audio_format}, {bits}-bit; the node reads 16-bit PCM."
                )
            end = len(data) if size == UNKNOWN_SIZE else min(start + size, len(data))
            pcm = data[start:end]
            whole = len(pcm) - len(pcm) % (2 * channels)
            if whole == 0:
                raise ValueError("Paxa's WAV holds no audio.")
            return channels, rate, pcm[:whole]
        if size == UNKNOWN_SIZE:
            break
        position = start + size + (size & 1)
    raise ValueError("Paxa's WAV has no audio data.")


def _resample(waveform: torch.Tensor, source: int, target: int) -> torch.Tensor:
    try:
        from comfy.audio import resample  # ComfyUI 0.40 and later
    except ImportError:
        pass
    else:
        return resample(waveform, source, target)
    try:
        from torchaudio.functional import resample
    except ImportError:
        raise ValueError(
            "The recording is too long to send at its sample rate, and this ComfyUI has no "
            "resampler. Trim it with Trim Audio Duration."
        ) from None
    return resample(waveform, source, target)
