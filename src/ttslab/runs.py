"""Erzeugte Audios samt Parametern ablegen: data/outputs/<run_id>.wav + <run_id>.json."""

import json
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Run:
    id: str
    engine_id: str
    engine_label: str
    model_id: str
    text: str
    voice_slug: str | None
    voice_name: str
    language: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    sample_rate: int = 0
    seconds: float = 0.0
    generate_seconds: float = 0.0
    created_at: str = ""
    favorite: bool = False
    audio_path: str = ""


class RunStore:
    """CRUD auf einem Verzeichnis mit Generier-Ergebnissen."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def _audio_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.wav"

    def _new_id(self, engine_id: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        return f"{stamp}-{engine_id}"

    def add(
        self,
        audio: np.ndarray,
        sample_rate: int,
        engine_id: str,
        engine_label: str,
        model_id: str,
        text: str,
        voice_slug: str | None,
        voice_name: str,
        language: str,
        seed: int,
        params: dict[str, Any],
        generate_seconds: float,
    ) -> Run:
        run_id = self._new_id(engine_id)
        sf.write(self._audio_path(run_id), audio, sample_rate)
        run = Run(
            id=run_id,
            engine_id=engine_id,
            engine_label=engine_label,
            model_id=model_id,
            text=text,
            voice_slug=voice_slug,
            voice_name=voice_name,
            language=language,
            seed=seed,
            params=params,
            sample_rate=sample_rate,
            seconds=round(len(audio) / sample_rate, 2),
            generate_seconds=round(generate_seconds, 2),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            audio_path=str(self._audio_path(run_id)),
        )
        self._write_meta(run)
        return run

    def _write_meta(self, run: Run) -> None:
        meta = asdict(run)
        meta.pop("audio_path")
        self._meta_path(run.id).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, run_id: str) -> Run:
        meta_path = self._meta_path(run_id)
        if not meta_path.exists():
            raise KeyError(f"Unbekannter Lauf: {run_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return Run(audio_path=str(self._audio_path(run_id)), **meta)

    def list(
        self, engine_id: str | None = None, voice_slug: str | None = None
    ) -> list[Run]:
        runs = [self.get(path.stem) for path in self.root.glob("*.json")]
        if engine_id:
            runs = [run for run in runs if run.engine_id == engine_id]
        if voice_slug:
            runs = [run for run in runs if run.voice_slug == voice_slug]
        return sorted(runs, key=lambda run: run.id, reverse=True)

    def set_favorite(self, run_id: str, favorite: bool) -> Run:
        run = replace(self.get(run_id), favorite=favorite)
        self._write_meta(run)
        return run

    def delete(self, run_id: str) -> None:
        self._meta_path(run_id).unlink(missing_ok=True)
        self._audio_path(run_id).unlink(missing_ok=True)

    def export_zip(self, run_ids: Sequence[str], target: Path | str) -> Path:
        """Ausgewählte Läufe mit Audio und Metadaten als ZIP bündeln."""
        target = Path(target)
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for run_id in run_ids:
                run = self.get(run_id)
                archive.write(run.audio_path, f"{run_id}.wav")
                archive.write(self._meta_path(run_id), f"{run_id}.json")
        return target
