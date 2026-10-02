"""Build small WAV files in memory for the tests."""

import io
import wave
from pathlib import Path

import numpy as np


def make_wav(samples: np.ndarray, rate: int = 16_000, channels: int = 1, sample_width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(rate)
        if sample_width == 2:
            wav.writeframes(samples.astype("<i2").tobytes())
        else:
            wav.writeframes(samples.astype(np.uint8).tobytes())
    return buffer.getvalue()


def one_second_of_silence() -> bytes:
    return make_wav(np.zeros(16_000, dtype=np.int16))


# --- real speech -------------------------------------------------------------
# German sentences synthesized with Piper and the "Thorsten" voice
# (rhasspy/piper-voices de_DE-thorsten-medium; voice model MIT, Thorsten-Voice
# dataset CC0), converted to 16 kHz mono 16-bit. See tests/fixtures/README.md.
#   speech_message.wav  "Ich schreibe gerade eine Nachricht an Max."
#   speech_meeting.wav  "Wir treffen uns morgen um zehn."
#   speech_ja.wav       "Ja."
FIXTURES = Path(__file__).parent / "fixtures"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def fixture_samples(name: str) -> np.ndarray:
    """Float32 samples in [-1, 1], like app.audio.decode_wav returns."""
    with wave.open(str(FIXTURES / name), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def seconds(samples: np.ndarray) -> float:
    return len(samples) / 16_000


def room_noise(duration_s: float, level_db: float, seed: int = 0) -> np.ndarray:
    """Gaussian noise at a given RMS level, as float32 samples."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 10 ** (level_db / 20), int(16_000 * duration_s)).astype(np.float32)


def to_wav(samples: np.ndarray) -> bytes:
    return make_wav(np.clip(samples * 32768, -32768, 32767).astype(np.int16))
