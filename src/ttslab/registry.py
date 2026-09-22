"""Engine-Katalog. Modelle werden erst beim ersten Generieren geladen und bleiben im RAM."""

from collections.abc import Callable
from dataclasses import dataclass

from .engines.base import Engine


@dataclass(frozen=True)
class EngineEntry:
    id: str
    label: str
    factory: Callable[[], Engine]


class Registry:
    def __init__(self, entries: list[EngineEntry]):
        self._entries = {entry.id: entry for entry in entries}
        self._instances: dict[str, Engine] = {}

    @property
    def ids(self) -> list[str]:
        return list(self._entries)

    def choices(self) -> list[tuple[str, str]]:
        """Für Gradio-Auswahlfelder: (Beschriftung, Wert)."""
        return [(entry.label, entry.id) for entry in self._entries.values()]

    def get(self, engine_id: str) -> Engine:
        if engine_id not in self._entries:
            raise KeyError(f"Unbekannte Engine: {engine_id}")
        if engine_id not in self._instances:
            self._instances[engine_id] = self._entries[engine_id].factory()
        return self._instances[engine_id]

    def is_loaded(self, engine_id: str) -> bool:
        return engine_id in self._instances


def default_registry() -> Registry:
    def chatterbox() -> Engine:
        from .engines.chatterbox import ChatterboxEngine

        return ChatterboxEngine()

    def qwen3() -> Engine:
        from .engines.qwen3 import Qwen3Engine

        return Qwen3Engine()

    def voxcpm2() -> Engine:
        from .engines.voxcpm2 import VoxCPM2Engine

        return VoxCPM2Engine()

    return Registry(
        [
            EngineEntry("qwen3", "Qwen3-TTS 1.7B", qwen3),
            EngineEntry("voxcpm2", "VoxCPM2", voxcpm2),
            EngineEntry("chatterbox", "Chatterbox Multilingual", chatterbox),
        ]
    )
