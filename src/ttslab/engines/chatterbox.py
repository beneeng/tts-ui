"""Chatterbox Multilingual (Resemble AI) über PyTorch/MPS."""

import copy
import time
from typing import Any

from .base import GenerationResult, ParamSpec

MODEL_ID = "ResembleAI/chatterbox (multilingual)"

PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "exaggeration",
        "Emotion-Exaggeration",
        "slider",
        0.5,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        help="0 = neutral, 1 = übertrieben",
    ),
    ParamSpec(
        "cfg_weight",
        "CFG-Gewicht",
        "slider",
        0.5,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        help="0 empfiehlt Resemble AI, wenn die Stimmprobe in einer anderen Sprache ist",
    ),
    ParamSpec("temperature", "Temperatur", "slider", 0.8, minimum=0.1, maximum=1.5, step=0.05),
    ParamSpec("top_p", "top_p", "slider", 1.0, minimum=0.1, maximum=1.0, step=0.05),
    ParamSpec("min_p", "min_p", "slider", 0.05, minimum=0.0, maximum=0.5, step=0.01),
    ParamSpec(
        "repetition_penalty",
        "Wiederholungs-Strafe",
        "slider",
        2.0,
        minimum=1.0,
        maximum=3.0,
        step=0.1,
    ),
)


class ChatterboxEngine:
    id = "chatterbox"
    label = "Chatterbox Multilingual"
    model_id = MODEL_ID
    needs_ref_text = False
    params = PARAMS

    def __init__(self, device: str = "mps"):
        self.device = device
        self._model = None
        self._builtin_conds = None

    def load(self) -> None:
        if self._model is not None:
            return
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        # Die Library überschreibt conds beim Klonen dauerhaft, deshalb das Original sichern.
        self._builtin_conds = copy.deepcopy(self._model.conds)

    def generate(
        self,
        text: str,
        ref_audio: str | None,
        ref_text: str | None,
        language: str,
        seed: int,
        **params: Any,
    ) -> GenerationResult:
        import torch

        self.load()
        torch.manual_seed(seed)
        if ref_audio is None:
            self._model.conds = copy.deepcopy(self._builtin_conds)

        started = time.perf_counter()
        wav = self._model.generate(
            text=text,
            language_id=language,
            audio_prompt_path=ref_audio,
            **params,
        )
        seconds = time.perf_counter() - started
        return GenerationResult(
            sample_rate=self._model.sr,
            audio=wav.squeeze(0).cpu().numpy(),
            seconds=seconds,
        )

    @staticmethod
    def languages() -> dict[str, str]:
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES

        return dict(SUPPORTED_LANGUAGES)
