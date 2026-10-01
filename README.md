# whisper-service

Local speech-to-text server for push-to-talk dictation, running whisper `large-v3-turbo` on the Apple Silicon GPU via [MLX](https://github.com/ml-explore/mlx).
It replaces the shared gpuserver whisper service with one that lives on the Mac itself, so dictation works offline, without Tailscale, and never depends on another machine being up.

It speaks exactly the gpuserver's HTTP contract, so [`mac-voice-dictation`](https://github.com/gapsong/mac-voice-dictation) and [`linux-voice-dictation`](https://github.com/gapsong/linux-voice-dictation) work against it by changing only their server URL.

## Install

Needs macOS on Apple Silicon and [uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:gapsong/whisper-service.git
cd whisper-service
scripts/install.sh
```

This syncs the pinned dependencies from `uv.lock` and installs a LaunchAgent (`com.gapsong.whisper-service`) that starts the server at login and restarts it if it crashes.
It listens on `http://127.0.0.1:9876` only - nothing outside the Mac can reach it.
Run `scripts/install.sh` again after a `git pull` to update; `scripts/uninstall.sh` removes the LaunchAgent.

The first `/start` downloads the model (~1.6 GB) to `~/.cache/huggingface`, which takes a minute or two once.
Logs go to `~/Library/Logs/whisper-service.log`.

## Measured on an M5 Pro (48 GB)

| What | Time |
| --- | --- |
| Wake up: load the model from disk plus warm-up | 0.5 - 0.7 s |
| Transcribe 4.4 s of German speech | 0.18 - 0.20 s |
| Memory while loaded / asleep | 2.7 GB / 0.17 GB |

The clients fire `/start` the moment the dictation key goes down, so the wake-up happens while you are still speaking.

## API

| Endpoint | Method | Notes |
| --- | --- | --- |
| `/health` | GET | `{"model", "state", "ready"}`, `state` is `sleeping` or `ready`. |
| `/start` | POST | Starts loading the model and returns the same body at once; it never blocks. |
| `/transcribe` | POST | Body: 16 kHz mono 16-bit WAV, header `X-Language: de`, `en` or `auto` -> `{"text", "language", "ms"}`. |

While the model is still loading, `/transcribe` answers HTTP 503.
Both clients poll `/health` after `/start` and retry on 500/503, so a cold start just works.
A WAV in any other format is refused with HTTP 400 instead of being guessed at.

`state` is only ever `sleeping` or `ready`, never `loading`: the Mac client decodes it into a two-value enum.
A model that is loading is reported as `sleeping` with `ready: false`.

## Design

**Asleep until needed.**
The model takes no memory until a client calls `/start`.
After 30 minutes without a transcription it is unloaded again (set `WHISPER_IDLE_UNLOAD_SECONDS` to change that).

**One worker thread for all model work.**
MLX binds its GPU stream to the thread that created it, so load, transcribe and unload all run on one dedicated thread (`app/engine.py`).
That also queues concurrent requests one after another, which is what a single GPU wants.

**Warm-up on load.**
MLX compiles its GPU kernels on first use, which made the first request after a load take 2.7 s instead of 0.2 s.
Loading therefore ends with a dummy transcription of one second of silence (`app/mlx_backend.py`).

**Raw text out.**
The server returns whisper's text as is.
Filtering whisper's silence artifacts ("Untertitelung des ZDF, 2020") stays in the clients, where it already lives.

## Layout

```
app/audio.py         WAV bytes -> float samples, strict 16 kHz / 16-bit
app/engine.py        sleeping / loading / ready state, idle unload, the worker thread
app/mlx_backend.py   whisper on MLX (load, unload, transcribe)
app/server.py        the HTTP contract (FastAPI)
scripts/install.sh   LaunchAgent install / update
tests/               contract and audio tests with a fake model
```

## Tests

```sh
uv run pytest
```

The tests use a fake model, so they run in about a second without a GPU.
They cover the full contract: asleep at boot, `/start` not blocking, 503 while loading, the language header, bad audio, idle unload and wake-up, and recovery from a failed load.

Run the server in the foreground for manual checks:

```sh
uv run uvicorn --factory app.server:create_default_app --host 127.0.0.1 --port 9876
```
