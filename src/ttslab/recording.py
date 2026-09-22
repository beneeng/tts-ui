"""Aufnahmen aus dem Browser entgegennehmen und in WAV umwandeln.

Der Browser schickt die Rohaufnahme (meist WebM/Opus) unverändert an den Server.
Gradios eigener Aufnahme-Baustein rechnet stattdessen im Browser und braucht dafür in
Brave quadratisch lange: 20 Sekunden Aufnahme kosteten dort 193 Sekunden.
"""

import shutil
import subprocess
import uuid
from pathlib import Path

DEFAULT_SAMPLE_RATE = 24000


class FfmpegMissing(RuntimeError):
    """ffmpeg ist nicht installiert."""


class ConversionFailed(RuntimeError):
    """ffmpeg konnte die Aufnahme nicht lesen."""


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def convert_to_wav(
    source: Path | str, target: Path | str, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> Path:
    """Beliebiges Audioformat in Mono-WAV umwandeln."""
    executable = ffmpeg_path()
    if executable is None:
        raise FfmpegMissing(
            "ffmpeg wird zum Umwandeln der Aufnahme gebraucht: brew install ffmpeg"
        )

    result = subprocess.run(
        [executable, "-y", "-loglevel", "error", "-i", str(source),
         "-ac", "1", "-ar", str(sample_rate), str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ConversionFailed(result.stderr.strip()[:300] or "unbekannter ffmpeg-Fehler")
    return Path(target)


def save_recording(
    data: bytes,
    incoming_dir: Path | str,
    suffix: str = ".webm",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """Rohaufnahme ablegen, nach WAV wandeln und den Pfad zurückgeben."""
    if not data:
        raise ValueError("Die Aufnahme ist leer.")

    incoming = Path(incoming_dir)
    incoming.mkdir(parents=True, exist_ok=True)
    raw = incoming / f"{uuid.uuid4().hex}{suffix}"
    raw.write_bytes(data)
    try:
        return convert_to_wav(raw, raw.with_suffix(".wav"), sample_rate=sample_rate)
    finally:
        raw.unlink(missing_ok=True)
