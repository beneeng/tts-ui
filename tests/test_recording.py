import subprocess

import numpy as np
import pytest
import soundfile as sf

from ttslab.recording import (
    ConversionFailed,
    FfmpegMissing,
    convert_to_wav,
    ffmpeg_path,
    save_recording,
)

needs_ffmpeg = pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg nicht installiert")


@pytest.fixture
def opus_recording(tmp_path):
    """Eine echte WebM/Opus-Datei, wie sie der Browser schickt."""
    source = tmp_path / "quelle.wav"
    t = np.linspace(0, 2.0, 2 * 48000, dtype="float32")
    sf.write(source, 0.2 * np.sin(2 * np.pi * 220 * t), 48000)
    target = tmp_path / "aufnahme.webm"
    subprocess.run(
        [ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(source), "-c:a", "libopus", str(target)],
        check=True,
    )
    return target.read_bytes()


@needs_ffmpeg
def test_save_recording_writes_mono_wav(tmp_path, opus_recording):
    path = save_recording(opus_recording, tmp_path / "incoming", sample_rate=24000)

    info = sf.info(str(path))
    assert path.suffix == ".wav"
    assert info.channels == 1
    assert info.samplerate == 24000
    assert 1.8 < info.frames / info.samplerate < 2.2


@needs_ffmpeg
def test_raw_upload_is_cleaned_up(tmp_path, opus_recording):
    incoming = tmp_path / "incoming"

    save_recording(opus_recording, incoming)

    assert [p.suffix for p in incoming.iterdir()] == [".wav"]


def test_empty_recording_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="leer"):
        save_recording(b"", tmp_path)


@needs_ffmpeg
def test_garbage_recording_reports_conversion_error(tmp_path):
    with pytest.raises(ConversionFailed):
        save_recording(b"kein audio, nur text", tmp_path)


def test_missing_ffmpeg_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr("ttslab.recording.ffmpeg_path", lambda: None)

    with pytest.raises(FfmpegMissing, match="brew install ffmpeg"):
        convert_to_wav(tmp_path / "egal.webm", tmp_path / "egal.wav")
