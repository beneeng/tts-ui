"""Reine Hilfsfunktionen für die Oberfläche: Auswahllisten, Tabellenzeilen, Parameter-Bündelung."""

from typing import Any

from .runs import Run
from .voices import VoiceSample

# Kanonische Sprachcodes der App. Die Engines übersetzen sie in ihre eigene Schreibweise.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("Deutsch", "de"),
    ("Englisch", "en"),
    ("Französisch", "fr"),
    ("Spanisch", "es"),
    ("Italienisch", "it"),
    ("Niederländisch", "nl"),
    ("Portugiesisch", "pt"),
    ("Polnisch", "pl"),
    ("Russisch", "ru"),
)

NO_VOICE = "__keine__"

RUN_COLUMNS = ("Zeit", "Engine", "Stimme", "Seed", "Dauer", "Rechenzeit", "★", "Text", "ID")
VOICE_COLUMNS = ("Name", "Dauer", "Sprache", "Transkript", "Slug")


def voice_choices(samples: list[VoiceSample]) -> list[tuple[str, str]]:
    """Auswahlliste der Stimmproben, immer mit der Option ohne Stimmprobe."""
    return [("ohne Stimmprobe", NO_VOICE)] + [
        (f"{sample.name} ({sample.seconds:.0f} s)", sample.slug) for sample in samples
    ]


def voice_rows(samples: list[VoiceSample]) -> list[list[Any]]:
    return [
        [sample.name, f"{sample.seconds:.1f} s", sample.language, shorten(sample.transcript), sample.slug]
        for sample in samples
    ]


def run_rows(runs: list[Run]) -> list[list[Any]]:
    return [
        [
            run.created_at[11:19],
            run.engine_label,
            run.voice_name,
            run.seed,
            f"{run.seconds:.1f} s",
            f"{run.generate_seconds:.1f} s",
            "★" if run.favorite else "",
            shorten(run.text),
            run.id,
        ]
        for run in runs
    ]


def shorten(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def group_params(
    keys: list[tuple[str, str]], values: list[Any]
) -> dict[str, dict[str, Any]]:
    """Flache Regler-Werte der UI nach Engine bündeln.

    keys ist die Zuordnung Komponente -> (engine_id, parameter_name) in derselben
    Reihenfolge wie values.
    """
    if len(keys) != len(values):
        raise ValueError("Zu den Parameter-Schlüsseln fehlen Werte.")
    grouped: dict[str, dict[str, Any]] = {}
    for (engine_id, name), value in zip(keys, values):
        grouped.setdefault(engine_id, {})[name] = value
    return grouped


def voice_slug_or_none(choice: str | None) -> str | None:
    return None if choice in (None, "", NO_VOICE) else choice


def run_label(run: Run) -> str:
    """Überschrift über einem Player im Vergleich."""
    params = ", ".join(f"{name}={value}" for name, value in sorted(run.params.items()))
    return (
        f"**{run.engine_label}** · Stimme: {run.voice_name} · Seed {run.seed} · "
        f"{run.seconds:.1f} s Audio in {run.generate_seconds:.1f} s\n\n`{params}`"
    )
