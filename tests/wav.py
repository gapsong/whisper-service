"""Build small WAV files in memory for the tests."""

import io
import wave

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
