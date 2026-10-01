"""Whisper on the Apple Silicon GPU, via mlx-whisper.

mlx-whisper keeps the loaded model in its own module-level cache
(`ModelHolder`). Loading means filling that cache; unloading means emptying
it and handing the freed GPU memory back to macOS.
"""

from typing import Optional

import mlx.core as mx
import mlx_whisper
import numpy as np
from mlx_whisper.transcribe import ModelHolder

DTYPE = mx.float16


class MlxWhisperBackend:
    def __init__(self, model_repo: str):
        self.model_repo = model_repo

    def load(self) -> None:
        ModelHolder.get_model(self.model_repo, DTYPE)
        # MLX compiles its GPU kernels on first use. Measured: the first real
        # request took 2.7 s, later ones 0.18 s. A dummy run on one second of
        # silence pays that cost here, while the user is still speaking.
        self.transcribe(np.zeros(16_000, dtype=np.float32), language="de")

    def unload(self) -> None:
        ModelHolder.model = None
        ModelHolder.model_path = None
        mx.clear_cache()

    def transcribe(self, samples: np.ndarray, language: Optional[str]) -> dict:
        result = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self.model_repo,
            language=language,
            task="transcribe",
            # Each utterance stands alone. Feeding the previous window's text
            # back in only helps long recordings and invites repetition loops.
            condition_on_previous_text=False,
            verbose=None,
        )
        return {"text": result["text"].strip(), "language": result.get("language")}
