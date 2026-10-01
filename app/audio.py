"""Turn the WAV bytes a client uploads into the float samples whisper expects.

Both clients (mac-voice-dictation, linux-voice-dictation) send 16 kHz, mono,
16-bit PCM. That is exactly whisper's native input, so we only accept that
format and reject anything else with a clear message instead of guessing.
"""

import io
import wave

import numpy as np

SAMPLE_RATE = 16_000


class AudioError(ValueError):
    """The upload is not a WAV file we can transcribe."""


def decode_wav(data: bytes) -> np.ndarray:
    """Return the WAV's samples as float32 in [-1, 1], one channel."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioError(f"not a readable WAV file: {exc}") from exc

    if sample_width != 2:
        raise AudioError(f"expected 16-bit PCM, got {sample_width * 8}-bit")
    if rate != SAMPLE_RATE:
        raise AudioError(f"expected {SAMPLE_RATE} Hz, got {rate} Hz")

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples
