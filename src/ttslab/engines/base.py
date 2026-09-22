"""Engine-Abstraktion: Jede TTS-Engine deklariert ihre Parameter selbst."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

ParamKind = Literal["slider", "dropdown", "checkbox", "number", "text"]


@dataclass(frozen=True)
class ParamSpec:
    """Beschreibt einen Engine-Parameter, aus dem die UI ihre Komponente baut."""

    name: str
    label: str
    kind: ParamKind
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[tuple[str, Any], ...] | None = None
    help: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    sample_rate: int
    audio: np.ndarray
    seconds: float


@runtime_checkable
class Engine(Protocol):
    """Gemeinsame Schnittstelle aller Engines."""

    id: str
    label: str
    model_id: str
    needs_ref_text: bool
    params: tuple[ParamSpec, ...]

    def load(self) -> None:
        """Modell in den Speicher laden. Mehrfachaufruf ist erlaubt."""

    def generate(
        self,
        text: str,
        ref_audio: str | None,
        ref_text: str | None,
        language: str,
        seed: int,
        **params: Any,
    ) -> GenerationResult:
        """Text in Mono-Audio umwandeln."""


def defaults(specs: tuple[ParamSpec, ...]) -> dict[str, Any]:
    return {spec.name: spec.default for spec in specs}


def clean_params(specs: tuple[ParamSpec, ...], values: dict[str, Any]) -> dict[str, Any]:
    """Nur Parameter durchlassen, die die Engine kennt, und Zahlen passend casten."""
    by_name = {spec.name: spec for spec in specs}
    cleaned: dict[str, Any] = {}
    for name, value in values.items():
        spec = by_name.get(name)
        if spec is None or value is None:
            continue
        if spec.kind in ("slider", "number") and isinstance(spec.default, int):
            cleaned[name] = int(value)
        elif spec.kind in ("slider", "number"):
            cleaned[name] = float(value)
        else:
            cleaned[name] = value
    return cleaned
