"""Stimmproben auf der Platte verwalten: data/voices/<slug>/audio.wav + meta.json."""

import json
import re
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

META_NAME = "meta.json"
AUDIO_NAME = "audio.wav"


@dataclass(frozen=True)
class VoiceSample:
    slug: str
    name: str
    transcript: str
    language: str
    seconds: float
    created_at: str
    note: str = ""
    audio_path: str = ""


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")
    return slug or "stimme"


class VoiceStore:
    """CRUD auf einem Verzeichnis mit Stimmproben."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, slug: str) -> Path:
        return self.root / slug

    def _free_slug(self, name: str) -> str:
        base = slugify(name)
        slug = base
        counter = 2
        while self._dir(slug).exists():
            slug = f"{base}-{counter}"
            counter += 1
        return slug

    def add(
        self,
        audio_path: Path | str,
        name: str,
        transcript: str = "",
        language: str = "de",
        note: str = "",
    ) -> VoiceSample:
        info = sf.info(str(audio_path))
        slug = self._free_slug(name)
        target_dir = self._dir(slug)
        target_dir.mkdir(parents=True)
        audio, sample_rate = sf.read(str(audio_path), always_2d=False)
        if audio.ndim > 1:  # auf Mono mischen, die Engines erwarten einen Kanal
            audio = audio.mean(axis=1)
        sf.write(target_dir / AUDIO_NAME, audio, sample_rate)

        sample = VoiceSample(
            slug=slug,
            name=name.strip() or slug,
            transcript=transcript,
            language=language,
            seconds=round(info.frames / info.samplerate, 2),
            created_at=datetime.now(timezone.utc).isoformat(),  # mit Mikrosekunden, sortiert eindeutig
            note=note,
            audio_path=str(target_dir / AUDIO_NAME),
        )
        self._write_meta(sample)
        return sample

    def _write_meta(self, sample: VoiceSample) -> None:
        meta = asdict(sample)
        meta.pop("audio_path")
        (self._dir(sample.slug) / META_NAME).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, slug: str) -> VoiceSample:
        meta_path = self._dir(slug) / META_NAME
        if not meta_path.exists():
            raise KeyError(f"Unbekannte Stimmprobe: {slug}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return VoiceSample(audio_path=str(self._dir(slug) / AUDIO_NAME), **meta)

    def list(self) -> list[VoiceSample]:
        samples = [
            self.get(path.parent.name)
            for path in sorted(self.root.glob(f"*/{META_NAME}"))
        ]
        return sorted(samples, key=lambda s: s.created_at, reverse=True)

    def update(self, slug: str, **fields) -> VoiceSample:
        allowed = {"name", "transcript", "language", "note"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Nicht änderbare Felder: {sorted(unknown)}")
        sample = replace(self.get(slug), **fields)
        self._write_meta(sample)
        return sample

    def delete(self, slug: str) -> None:
        shutil.rmtree(self._dir(slug), ignore_errors=True)
