"""Owns the whisper model: loads it on demand and unloads it when idle.

The service boots asleep: the model costs no memory
until a client presses its dictation key and calls POST /start. After
`idle_unload_seconds` without a transcription the model is unloaded again.

All model work (load, transcribe, unload) runs on ONE dedicated worker thread.
MLX binds its GPU stream to the thread that created it, so touching the model
from FastAPI's thread pool would fail at random. One thread also means requests
are naturally queued one after another, which is what a single GPU wants.
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger(__name__)

SLEEPING = "sleeping"
LOADING = "loading"
READY = "ready"


class Backend(Protocol):
    """The part that actually runs whisper. Swapped for a fake in tests."""

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def transcribe(self, samples: np.ndarray, language: Optional[str]) -> dict:
        """Return {"text": str, "language": str | None}."""
        ...


class NotReady(Exception):
    """The model is not loaded yet. Loading has been started; try again soon."""


class Engine:
    def __init__(self, backend: Backend, idle_unload_seconds: float):
        self._backend = backend
        self._idle_unload_seconds = idle_unload_seconds
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._lock = threading.Lock()
        self._state = SLEEPING
        self._last_used = time.monotonic()
        self._idle_timer: Optional[threading.Timer] = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def start(self) -> str:
        """Begin loading the model if it is asleep. Never blocks. Returns the state."""
        with self._lock:
            self._last_used = time.monotonic()
            if self._state == SLEEPING:
                self._state = LOADING
                self._worker.submit(self._load)
            return self._state

    def transcribe(self, samples: np.ndarray, language: Optional[str]) -> dict:
        """Transcribe on the worker thread. Raises NotReady while the model loads."""
        if self.start() != READY:
            raise NotReady()
        future: Future = self._worker.submit(self._backend.transcribe, samples, language)
        result = future.result()
        with self._lock:
            self._last_used = time.monotonic()
        return result

    def shutdown(self) -> None:
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
        self._worker.shutdown(wait=False, cancel_futures=True)

    # --- runs on the worker thread -------------------------------------------

    def _load(self) -> None:
        try:
            self._backend.load()
        except Exception:
            # Back to sleeping, so the next /start simply tries again.
            log.exception("loading the whisper model failed")
            with self._lock:
                self._state = SLEEPING
            return
        with self._lock:
            self._state = READY
        self._schedule_idle_check(self._idle_unload_seconds)

    def _unload_if_idle(self) -> None:
        with self._lock:
            if self._state != READY:
                return
            idle_for = time.monotonic() - self._last_used
            if idle_for < self._idle_unload_seconds:
                remaining = self._idle_unload_seconds - idle_for
            else:
                remaining = None
                self._state = SLEEPING
        if remaining is not None:
            self._schedule_idle_check(remaining)
            return
        self._backend.unload()

    # --- idle timer ----------------------------------------------------------

    def _schedule_idle_check(self, delay: float) -> None:
        timer = threading.Timer(delay, lambda: self._worker.submit(self._unload_if_idle))
        timer.daemon = True
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._idle_timer = timer
        timer.start()
