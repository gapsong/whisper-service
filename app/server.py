"""HTTP API, identical to the old gpuserver whisper service.

mac-voice-dictation and linux-voice-dictation both speak this contract, so they
work against this server by changing only their server URL.

    GET  /health      -> {"model", "state", "ready"}     state: sleeping | ready
    POST /start       -> same body; starts loading the model, never blocks
    POST /transcribe  -> body: 16 kHz mono 16-bit WAV, header X-Language: de|en|auto
                      <- {"text", "language", "ms"}

While the model is still loading, /transcribe answers 503. Both clients poll
/health after /start and also retry 500/503, so a cold start just works.

`state` is only ever "sleeping" or "ready": the Mac client decodes it into a
two-value enum, so "loading" is reported as "sleeping" with ready=false.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.audio import AudioError, decode_wav
from app.engine import READY, SLEEPING, Backend, Engine, NotReady

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_IDLE_UNLOAD_SECONDS = 30 * 60


def create_app(backend: Backend, model_name: str, idle_unload_seconds: float) -> FastAPI:
    engine = Engine(backend, idle_unload_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        engine.shutdown()

    app = FastAPI(title="whisper-service", lifespan=lifespan)

    def health_body(state: str) -> dict:
        is_ready = state == READY
        return {"model": model_name, "state": READY if is_ready else SLEEPING, "ready": is_ready}

    @app.get("/health")
    def health() -> dict:
        return health_body(engine.state)

    @app.post("/start")
    def start() -> dict:
        return health_body(engine.start())

    @app.post("/transcribe")
    async def transcribe(request: Request, x_language: Optional[str] = Header(default=None)):
        language = None if x_language in (None, "", "auto") else x_language
        try:
            samples = decode_wav(await request.body())
        except AudioError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        started = time.monotonic()
        try:
            # Wait for the worker in a thread, so /health polls stay responsive.
            result = await run_in_threadpool(engine.transcribe, samples, language)
        except NotReady:
            return JSONResponse(status_code=503, content={"detail": "model is loading, retry"})
        elapsed_ms = round((time.monotonic() - started) * 1000)
        return {"text": result["text"], "language": result["language"], "ms": elapsed_ms}

    return app


def create_default_app() -> FastAPI:
    """Entry point for uvicorn (`--factory app.server:create_default_app`)."""
    from app.mlx_backend import MlxWhisperBackend

    model = os.environ.get("WHISPER_MODEL", DEFAULT_MODEL)
    idle = float(os.environ.get("WHISPER_IDLE_UNLOAD_SECONDS", DEFAULT_IDLE_UNLOAD_SECONDS))
    return create_app(MlxWhisperBackend(model), model, idle)
