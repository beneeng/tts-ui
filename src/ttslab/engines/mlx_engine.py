"""Gemeinsame Basis für alle Engines aus mlx-audio (Apple Silicon)."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .base import GenerationResult, ParamSpec


def _default_loader(model_id: str):
    from mlx_audio.tts.utils import load_model

    return load_model(model_id)


class MlxEngine:
    """generate() von mlx-audio liefert Segmente, die hier zu einem Audio zusammengefügt werden."""

    id = "mlx"
    label = "MLX"
    model_id = ""
    needs_ref_text = False
    needs_ref_audio = False
    params: tuple[ParamSpec, ...] = ()

    def __init__(self, model_id: str | None = None, loader: Callable[[str], Any] | None = None):
        self.model_id = model_id or self.model_id
        self._loader = loader or _default_loader
        self._model = None

    def load(self) -> None:
        if self._model is None:
            self._model = self._loader(self.model_id)

    def _language_kwargs(self, language: str) -> dict[str, Any]:
        """Kanonischen Sprachcode (de, en, …) in die Schreibweise der Engine übersetzen."""
        return {}

    def _seed(self, seed: int) -> None:
        import mlx.core as mx

        mx.random.seed(seed)

    def generate(
        self,
        text: str,
        ref_audio: str | None,
        ref_text: str | None,
        language: str,
        seed: int,
        **params: Any,
    ) -> GenerationResult:
        self.load()
        self._seed(seed)

        kwargs: dict[str, Any] = dict(params)
        kwargs.update(self._language_kwargs(language))
        kwargs.setdefault("verbose", False)  # sonst schreibt mlx-audio pro Segment ins Log
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text

        started = time.perf_counter()
        segments = list(self._model.generate(text=text, **kwargs))
        seconds = time.perf_counter() - started
        if not segments:
            raise RuntimeError(f"{self.label} hat kein Audio geliefert.")

        audio = np.concatenate([np.asarray(segment.audio, dtype="float32") for segment in segments])
        return GenerationResult(
            sample_rate=int(segments[0].sample_rate),
            audio=audio.reshape(-1),
            seconds=seconds,
        )
