"""Die Logik hinter dem Generieren-Knopf: ein Text, mehrere Engines, mehrere Takes."""

import random
from collections.abc import Iterator
from typing import Any

from .engines.base import clean_params
from .registry import Registry
from .runs import Run, RunStore
from .voices import VoiceStore

MAX_SEED = 2**31 - 1


def seeds_for(takes: int, base_seed: int | None) -> list[int]:
    """Feste Seeds bei Vorgabe, sonst zufällige. Takes bekommen immer verschiedene Seeds."""
    if base_seed is None:
        return random.sample(range(MAX_SEED), takes)
    return [base_seed + offset for offset in range(takes)]


def generate_runs(
    registry: Registry,
    voice_store: VoiceStore,
    run_store: RunStore,
    text: str,
    engine_ids: list[str],
    params_by_engine: dict[str, dict[str, Any]],
    voice_slug: str | None = None,
    language: str = "de",
    takes: int = 1,
    base_seed: int | None = None,
) -> Iterator[Run]:
    """Erzeugt pro Engine und Take einen Lauf und legt ihn ab. Liefert sie einzeln zurück."""
    if not text.strip():
        raise ValueError("Bitte Text eingeben.")
    if not engine_ids:
        raise ValueError("Bitte mindestens eine Engine auswählen.")
    if takes < 1:
        raise ValueError("Mindestens ein Take.")

    sample = voice_store.get(voice_slug) if voice_slug else None
    seeds = seeds_for(takes, base_seed)

    for engine_id in engine_ids:
        engine = registry.get(engine_id)
        if sample is None and (engine.needs_ref_audio or engine.needs_ref_text):
            raise ValueError(f"{engine.label} braucht eine Stimmprobe.")
        params = clean_params(engine.params, params_by_engine.get(engine_id, {}))

        for seed in seeds:
            result = engine.generate(
                text=text,
                ref_audio=sample.audio_path if sample else None,
                ref_text=sample.transcript if sample else None,
                language=language,
                seed=seed,
                **params,
            )
            yield run_store.add(
                audio=result.audio,
                sample_rate=result.sample_rate,
                engine_id=engine.id,
                engine_label=engine.label,
                model_id=engine.model_id,
                text=text,
                voice_slug=sample.slug if sample else None,
                voice_name=sample.name if sample else "ohne Stimmprobe",
                language=language,
                seed=seed,
                params=params,
                generate_seconds=result.seconds,
            )
