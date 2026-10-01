"""The HTTP contract both dictation clients depend on, with a fake model."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.engine import LOADING, READY, SLEEPING, Engine
from app.server import create_app
from tests.wav import fixture_bytes, one_second_of_silence

SPEECH = fixture_bytes("speech_message.wav")


class FakeBackend:
    """Stands in for whisper. Loading blocks until the test allows it."""

    def __init__(self):
        self.allow_load = threading.Event()
        self.fail_next_load = False
        self.loads = 0
        self.unloads = 0
        self.languages = []

    def load(self):
        self.allow_load.wait(timeout=5)
        if self.fail_next_load:
            self.fail_next_load = False
            raise RuntimeError("model download failed")
        self.loads += 1

    def unload(self):
        self.unloads += 1

    def transcribe(self, samples, language):
        self.languages.append(language)
        return {"text": "Hallo Welt", "language": language or "de"}


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


@pytest.fixture
def backend():
    return FakeBackend()


def make_client(backend, idle_unload_seconds=60.0):
    return TestClient(create_app(backend, "test-model", idle_unload_seconds))


def is_ready(client):
    return client.get("/health").json()["ready"]


def test_boots_asleep(backend):
    with make_client(backend) as client:
        assert client.get("/health").json() == {"model": "test-model", "state": "sleeping", "ready": False}
        assert backend.loads == 0


def test_start_does_not_block_and_reports_sleeping_while_loading(backend):
    with make_client(backend) as client:
        body = client.post("/start").json()
        # "loading" is not a state the Mac client can decode, so it must say sleeping.
        assert body == {"model": "test-model", "state": "sleeping", "ready": False}

        backend.allow_load.set()
        wait_for(lambda: is_ready(client))
        assert client.get("/health").json() == {"model": "test-model", "state": "ready", "ready": True}


def test_start_twice_loads_once(backend):
    backend.allow_load.set()
    with make_client(backend) as client:
        client.post("/start")
        client.post("/start")
        wait_for(lambda: is_ready(client))
        client.post("/start")
        assert backend.loads == 1


def test_transcribe_while_asleep_returns_503_and_starts_loading(backend):
    with make_client(backend) as client:
        response = client.post("/transcribe", content=SPEECH,
                               headers={"Content-Type": "audio/wav", "X-Language": "de"})
        assert response.status_code == 503

        backend.allow_load.set()
        wait_for(lambda: is_ready(client))


def test_transcribe_when_ready(backend):
    backend.allow_load.set()
    with make_client(backend) as client:
        client.post("/start")
        wait_for(lambda: is_ready(client))

        response = client.post("/transcribe", content=SPEECH,
                               headers={"Content-Type": "audio/wav", "X-Language": "de"})
        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "Hallo Welt"
        assert body["language"] == "de"
        assert isinstance(body["ms"], int)


@pytest.mark.parametrize("header, expected", [("de", "de"), ("en", "en"), ("auto", None), (None, None)])
def test_language_header(backend, header, expected):
    backend.allow_load.set()
    with make_client(backend) as client:
        client.post("/start")
        wait_for(lambda: is_ready(client))
        headers = {"Content-Type": "audio/wav"}
        if header is not None:
            headers["X-Language"] = header
        client.post("/transcribe", content=SPEECH, headers=headers)
        assert backend.languages == [expected]


def test_silence_is_empty_text_without_running_whisper(backend):
    backend.allow_load.set()
    with make_client(backend) as client:
        client.post("/start")
        wait_for(lambda: is_ready(client))

        response = client.post("/transcribe", content=one_second_of_silence(),
                               headers={"Content-Type": "audio/wav", "X-Language": "de"})
        assert response.status_code == 200
        assert response.json()["text"] == ""
        assert backend.languages == []  # whisper never ran


def test_silence_while_asleep_is_empty_text_not_503(backend):
    with make_client(backend) as client:
        response = client.post("/transcribe", content=one_second_of_silence(),
                               headers={"Content-Type": "audio/wav"})
        assert response.status_code == 200
        assert response.json()["text"] == ""


def test_bad_audio_is_400(backend):
    with make_client(backend) as client:
        response = client.post("/transcribe", content=b"not audio", headers={"Content-Type": "audio/wav"})
        assert response.status_code == 400
        assert "WAV" in response.json()["detail"]


def test_unloads_after_idle_and_wakes_again(backend):
    backend.allow_load.set()
    with make_client(backend, idle_unload_seconds=0.2) as client:
        client.post("/start")
        wait_for(lambda: is_ready(client))

        wait_for(lambda: backend.unloads == 1)
        assert client.get("/health").json()["state"] == "sleeping"

        client.post("/start")
        wait_for(lambda: is_ready(client))
        assert backend.loads == 2


def test_use_keeps_model_loaded(backend):
    backend.allow_load.set()
    with make_client(backend, idle_unload_seconds=0.5) as client:
        client.post("/start")
        wait_for(lambda: is_ready(client))
        for _ in range(4):
            time.sleep(0.2)
            client.post("/transcribe", content=SPEECH, headers={"Content-Type": "audio/wav"})
        assert backend.unloads == 0
        assert is_ready(client)


def test_failed_load_falls_back_to_sleeping_and_can_retry(backend):
    # Checked on the engine itself: /health reports "loading" as "sleeping"
    # too, so only the engine can tell a failed load from one in progress.
    backend.fail_next_load = True
    backend.allow_load.set()
    engine = Engine(backend, idle_unload_seconds=60.0)
    try:
        assert engine.start() == LOADING
        wait_for(lambda: engine.state == SLEEPING)
        assert backend.loads == 0

        engine.start()
        wait_for(lambda: engine.state == READY)
        assert backend.loads == 1
    finally:
        engine.shutdown()
