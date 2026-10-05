# whisper-service

Local speech-to-text for push-to-talk dictation on Apple Silicon.
It runs OpenAI's whisper `large-v3-turbo` on the Mac's own GPU via [MLX](https://github.com/ml-explore/mlx) and answers in about 0.2 seconds.
No cloud, no account, no other machine - dictation keeps working offline.

It is the backend for [mac-voice-dictation](https://github.com/gapsong/mac-voice-dictation): hold a key, speak, release, and the text appears at your cursor in any app.

## Quick start

You need a Mac with Apple Silicon (M1 or newer).

```sh
git clone https://github.com/gapsong/whisper-service.git
cd whisper-service
scripts/install.sh
```

That is all.
The script installs [uv](https://docs.astral.sh/uv/) if it is missing (with Homebrew if you have it, else with uv's official installer), then Python and the dependencies, sets the server up to start at login, and downloads the model once (~1.6 GB, a few minutes).
When it prints `ready`, install the [Mac app](https://github.com/gapsong/mac-voice-dictation) - it finds the server on its own.

The server listens on `http://127.0.0.1:9876` only, so nothing outside your Mac can reach it.

## Numbers

Measured on a MacBook Pro M5 Pro (48 GB):

| What | Time |
| --- | --- |
| Transcribe 4.4 s of German speech | 0.18 - 0.20 s |
| Wake up after a break (load model + warm-up) | 0.5 - 0.7 s |
| A tap without speech | 0.02 - 0.06 s, returns nothing |
| Memory while loaded / asleep | 2.7 GB / 0.17 GB |

The app wakes the server the moment the dictation key goes down, so the wake-up happens while you are still speaking.

## How it works

**Only speech reaches whisper.**
Whisper invents text for audio without speech.
A silent tap, room noise or a long pause comes back as "Vielen Dank." or "Thanks for watching!" - and whisper reports that with full confidence (`no_speech_prob` 0.0 on large-v3-turbo), so it cannot catch it itself.
Loudness cannot catch it either: a key click is almost as loud as speech.
[Silero VAD](https://github.com/snakers4/silero-vad) (voice activity detection) can.
On silence, room noise and key clicks it stayed at or below 0.38; on normal, quiet and noisy speech and a single short "Ja." it reached 0.96 - 1.00.
So `app/vad.py` first cuts every recording down to the speech:

- nobody spoke: the server answers with empty text at once, and whisper never runs
- silence before and after the speech is removed
- pauses longer than 0.7 s are shortened to 0.3 s

Whisper only ever sees speech, so it has no silence to fill.
Saying "Vielen Dank" still works - only silence is blocked.

**Asleep until needed.**
The model takes no memory until the app asks for it.
After 30 minutes without dictation it is unloaded again.

**Fast from the first word.**
MLX compiles its GPU kernels on first use, which made the first request after a load take 2.7 s instead of 0.2 s.
Loading therefore ends with a short dummy run, so that cost is paid while you are still speaking.

**One worker thread for the model.**
MLX ties its GPU stream to the thread that created it, so loading, transcribing and unloading all run on one dedicated thread (`app/engine.py`).
Requests from several apps queue up one after another, which is what a single GPU wants.

## API

Any client can use it; the protocol is three plain HTTP calls.

| Endpoint | Method | Notes |
| --- | --- | --- |
| `/health` | GET | `{"model", "state", "ready"}`; `state` is `sleeping` or `ready`. |
| `/start` | POST | Starts loading the model and returns the same body at once. |
| `/transcribe` | POST | Body: 16 kHz mono 16-bit WAV; header `X-Language: de`, `en` or `auto` -> `{"text", "language", "ms"}`. |

A client calls `/start` when recording begins, polls `/health` until `ready`, then sends the audio.
While the model is still loading, `/transcribe` answers HTTP 503, so a client can simply retry.
Audio in any other format is refused with HTTP 400 rather than guessed at.

Try it by hand:

```sh
curl -X POST http://127.0.0.1:9876/start
curl -X POST -H 'Content-Type: audio/wav' -H 'X-Language: de' \
     --data-binary @tests/fixtures/speech_message.wav http://127.0.0.1:9876/transcribe
```

Clients that speak this protocol:

- [mac-voice-dictation](https://github.com/gapsong/mac-voice-dictation) - macOS menu-bar app
- [linux-voice-dictation](https://github.com/gapsong/linux-voice-dictation) - Linux / Wayland daemon (needs the server reachable over the network)

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Is it running? | `curl http://127.0.0.1:9876/health` |
| Something fails | read `~/Library/Logs/whisper-service.log` |
| Restart it | `scripts/install.sh` (safe to run any time) |
| Update | `git pull && scripts/install.sh` |
| Remove it | `scripts/uninstall.sh`, then delete the folder; the model lives in `~/.cache/huggingface` |

## Development

```sh
uv run pytest
```

The tests replace whisper with a fake, so they run in about two seconds without a GPU; the VAD runs for real on recorded speech (`tests/fixtures/`).
They cover the whole protocol, and that silence, noise and key clicks never reach whisper while normal, quiet and short speech always does.

Run the server in the foreground:

```sh
uv run uvicorn --factory app.server:create_default_app --host 127.0.0.1 --port 9876
```

`WHISPER_MODEL` (any [mlx-community whisper model](https://huggingface.co/mlx-community)) and `WHISPER_IDLE_UNLOAD_SECONDS` change the model and the idle timeout for such a run.

```
app/audio.py         WAV bytes -> float samples, strict 16 kHz / 16-bit
app/vad.py           keep only the speech (Silero VAD); empty if nobody spoke
app/engine.py        sleeping / loading / ready, idle unload, the worker thread
app/mlx_backend.py   whisper on MLX (load, unload, transcribe)
app/server.py        the HTTP API (FastAPI)
scripts/install.sh   install, update and restart the LaunchAgent
tests/               API, audio and VAD tests
```

## License

[MIT](LICENSE).
