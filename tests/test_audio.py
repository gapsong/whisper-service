import numpy as np
import pytest

from app.audio import AudioError, decode_wav
from tests.wav import make_wav


def test_mono_16bit_is_scaled_to_minus_one_to_one():
    samples = decode_wav(make_wav(np.array([0, 16384, -32768], dtype=np.int16)))
    assert samples.dtype == np.float32
    assert samples.tolist() == [0.0, 0.5, -1.0]


def test_stereo_is_mixed_down_to_mono():
    # Interleaved left/right frames: (1000, 3000) and (-2000, 0).
    wav = make_wav(np.array([1000, 3000, -2000, 0], dtype=np.int16), channels=2)
    assert decode_wav(wav).tolist() == pytest.approx([2000 / 32768, -1000 / 32768])


def test_wrong_sample_rate_is_rejected():
    with pytest.raises(AudioError, match="16000 Hz"):
        decode_wav(make_wav(np.zeros(10, dtype=np.int16), rate=44_100))


def test_8bit_audio_is_rejected():
    with pytest.raises(AudioError, match="16-bit"):
        decode_wav(make_wav(np.zeros(10), sample_width=1))


def test_garbage_is_rejected():
    with pytest.raises(AudioError, match="not a readable WAV"):
        decode_wav(b"definitely not a wav file")


def test_empty_body_is_rejected():
    with pytest.raises(AudioError):
        decode_wav(b"")
