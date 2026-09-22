"""Die HTTP-Route, an die der Recorder im Browser seine Rohaufnahme schickt."""

import importlib
import subprocess

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from ttslab.recording import ffmpeg_path

needs_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg nicht installiert")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TTSLAB_DATA", str(tmp_path))
    import app as app_module

    app_module = importlib.reload(app_module)
    return TestClient(app_module.api), tmp_path


@pytest.fixture
def opus_blob(tmp_path):
    source = tmp_path / "ton.wav"
    t = np.linspace(0, 3.0, 3 * 48000, dtype="float32")
    sf.write(source, 0.2 * np.sin(2 * np.pi * 180 * t), 48000)
    target = tmp_path / "aufnahme.webm"
    subprocess.run(
        [ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(source), "-c:a", "libopus", str(target)],
        check=True,
    )
    return target.read_bytes()


@needs_ffmpeg
def test_recording_is_stored_as_wav(client, opus_blob):
    http, data_dir = client

    response = http.post(
        "/ttslab/record", content=opus_blob, headers={"Content-Type": "audio/webm"}
    )

    assert response.status_code == 200
    path = response.json()["path"]
    assert path.startswith(str(data_dir))
    info = sf.info(path)
    assert info.channels == 1
    assert 2.8 < info.frames / info.samplerate < 3.2


def test_empty_recording_gives_400(client):
    http, _ = client

    response = http.post("/ttslab/record", content=b"")

    assert response.status_code == 400
    assert "leer" in response.json()["detail"]


@needs_ffmpeg
def test_unreadable_recording_gives_400(client):
    http, _ = client

    response = http.post("/ttslab/record", content=b"das ist kein audio")

    assert response.status_code == 400
    assert "nicht lesbar" in response.json()["detail"]


def test_missing_ffmpeg_gives_503(client, opus_blob, monkeypatch):
    http, _ = client
    monkeypatch.setattr("ttslab.recording.ffmpeg_path", lambda: None)

    response = http.post("/ttslab/record", content=opus_blob)

    assert response.status_code == 503
    assert "ffmpeg" in response.json()["detail"]
