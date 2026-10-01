"""keep_speech with the real Silero model: noise must vanish, speech must stay."""

import numpy as np
import pytest

from app.vad import keep_speech
from tests.wav import fixture_samples, room_noise, seconds

MESSAGE = fixture_samples("speech_message.wav")
MEETING = fixture_samples("speech_meeting.wav")
JA = fixture_samples("speech_ja.wav")


def key_clicks(duration_s: float) -> np.ndarray:
    """Room noise with a sharp 15 ms click at the start and the end, like a key press."""
    samples = room_noise(duration_s, -55)
    click = np.random.default_rng(1).normal(0, 0.15, 240).astype(np.float32) * np.hanning(240)
    samples[:240] += click
    samples[-240:] += click
    return samples


# --- nobody spoke: nothing may reach whisper ----------------------------------

@pytest.mark.parametrize("name, samples", [
    ("digital silence", np.zeros(8_000, dtype=np.float32)),
    ("quiet room", room_noise(0.8, -60)),
    ("room noise", room_noise(0.8, -45)),
    ("loud room", room_noise(0.8, -30)),
    ("key clicks", key_clicks(0.8)),
    ("long quiet hold", room_noise(5.0, -55)),
])
def test_no_speech_gives_nothing(name, samples):
    assert keep_speech(samples).size == 0


def test_too_short_to_judge_gives_nothing():
    assert keep_speech(np.zeros(100, dtype=np.float32)).size == 0


# --- someone spoke: the speech must survive ----------------------------------

@pytest.mark.parametrize("name, samples", [
    ("normal speech", MESSAGE),
    ("quiet speech (-26 dB)", MESSAGE * 0.05),
    ("speech in room noise", MESSAGE + room_noise(seconds(MESSAGE), -45)),
    ("one short word", JA),
    ("one short quiet word", JA * 0.1),
])
def test_speech_is_kept(name, samples):
    kept = keep_speech(samples)
    assert kept.size > 0
    assert kept.size <= samples.size  # never duplicated


def test_silence_around_speech_is_trimmed():
    recording = np.concatenate([room_noise(2.0, -55), MESSAGE, room_noise(3.0, -55)])
    kept = keep_speech(recording)
    # 5 s of silence went in; at most the padding (2 x 200 ms) may come out.
    assert seconds(kept) <= seconds(MESSAGE) + 0.5
    assert seconds(kept) >= seconds(MESSAGE) - 0.5


def test_long_pause_is_shortened_but_both_sentences_stay():
    recording = np.concatenate([MESSAGE, room_noise(4.0, -55), MEETING])
    kept = keep_speech(recording)
    speech_only = seconds(MESSAGE) + seconds(MEETING)
    # The 4 s pause shrinks to about PAUSE_MS plus padding.
    assert seconds(kept) <= speech_only + 1.2
    assert seconds(kept) >= speech_only - 0.5


def test_output_is_float32():
    assert keep_speech(MESSAGE).dtype == np.float32
