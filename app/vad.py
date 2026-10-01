"""Keep only the parts of a recording where someone is speaking.

Whisper invents text for audio without speech. Handed silence, room noise or a
long pause it answers with subtitle boilerplate, most often "Vielen Dank." -
and it reports that with full confidence (no_speech_prob 0.0 on
large-v3-turbo), so whisper itself cannot tell. Loudness cannot tell either:
a key click is nearly as loud as speech.

Silero VAD (voice activity detection) can. Measured on silence, room noise
and key clicks it never went above 0.38; on normal, quiet and noisy speech
and on a single short "Ja." it reached 0.96 - 1.00.

So before whisper runs, `keep_speech` cuts the recording down to the speech:

  - nobody spoke          -> empty array; the server answers "" right away
  - silence before/after  -> removed
  - a long pause          -> shortened to PAUSE_MS of silence

Whisper then only ever sees speech and has no silence to fill.
"""

import numpy as np
from silero_vad_lite import SileroVAD

SAMPLE_RATE = 16_000

# A window counts as speech above this probability (Silero's usual cut-off).
SPEECH_THRESHOLD = 0.5
# Less speech than this in the whole recording counts as "nobody spoke".
# A quiet, single "Ja." measured 224 ms, so 100 ms keeps short words safe.
MIN_SPEECH_MS = 100
# Kept around each region, so word onsets and endings are not clipped.
PAD_MS = 200
# Padded regions closer than this are one region. Together with the padding,
# gaps in the speech of up to 2 * PAD_MS + MERGE_GAP_MS = 700 ms (between
# words, a quick breath) stay untouched; only longer pauses get shortened.
MERGE_GAP_MS = 300
# Silence placed between regions that were further apart than MERGE_GAP_MS.
PAUSE_MS = 300


def _ms_to_samples(ms: int) -> int:
    return SAMPLE_RATE * ms // 1000


def speech_probabilities(samples: np.ndarray) -> tuple[np.ndarray, int]:
    """Speech probability per VAD window, and the window size in samples.

    A fresh SileroVAD per recording: the model is stateful (recurrent), and a
    new instance costs ~20 ms, so no state ever leaks from one request into
    the next.
    """
    vad = SileroVAD(SAMPLE_RATE)
    window = vad.window_size_samples
    samples = samples.astype(np.float32, copy=False)
    probabilities = [
        vad.process(np.ascontiguousarray(samples[start:start + window]))
        for start in range(0, len(samples) - window + 1, window)
    ]
    return np.array(probabilities, dtype=np.float32), window


def speech_regions(samples: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) sample ranges that contain speech, merged and padded."""
    probabilities, window = speech_probabilities(samples)
    is_speech = probabilities > SPEECH_THRESHOLD
    if is_speech.sum() * window < _ms_to_samples(MIN_SPEECH_MS):
        return []

    # Pad every speech window first, then merge: merging before padding could
    # leave padded neighbours overlapping, and that audio would be sent twice.
    pad = _ms_to_samples(PAD_MS)
    merge_gap = _ms_to_samples(MERGE_GAP_MS)
    regions: list[tuple[int, int]] = []
    for index in np.flatnonzero(is_speech):
        start = max(0, index * window - pad)
        end = min(len(samples), (index + 1) * window + pad)
        if regions and start <= regions[-1][1] + merge_gap:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))
    return regions


def keep_speech(samples: np.ndarray) -> np.ndarray:
    """The recording with everything but speech removed; empty if nobody spoke."""
    regions = speech_regions(samples)
    if not regions:
        return np.zeros(0, dtype=np.float32)

    pause = np.zeros(_ms_to_samples(PAUSE_MS), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for start, end in regions:
        if pieces:
            pieces.append(pause)
        pieces.append(samples[start:end].astype(np.float32, copy=False))
    return np.concatenate(pieces)


def warm_up() -> None:
    """Load the VAD library once at startup; the very first load takes ~1 s."""
    SileroVAD(SAMPLE_RATE)
