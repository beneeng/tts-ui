"""Qwen3-TTS (Alibaba, Apache 2.0) über mlx-audio."""

from typing import Any

from .base import ParamSpec
from .mlx_engine import MlxEngine

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

# Qwen erwartet ausgeschriebene Sprachnamen, nicht ISO-Codes.
LANGUAGE_NAMES = {
    "auto": "auto",
    "de": "german",
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
}


class Qwen3Engine(MlxEngine):
    id = "qwen3"
    label = "Qwen3-TTS 1.7B"
    model_id = MODEL_ID
    needs_ref_text = False
    params = (
        ParamSpec("temperature", "Temperatur", "slider", 0.9, minimum=0.1, maximum=1.5, step=0.05),
        ParamSpec("top_p", "top_p", "slider", 1.0, minimum=0.1, maximum=1.0, step=0.05),
        ParamSpec("top_k", "top_k", "slider", 50, minimum=1, maximum=100, step=1),
        ParamSpec(
            "repetition_penalty",
            "Wiederholungs-Strafe",
            "slider",
            1.05,
            minimum=1.0,
            maximum=2.0,
            step=0.05,
        ),
        ParamSpec("speed", "Tempo", "slider", 1.0, minimum=0.5, maximum=2.0, step=0.05),
        ParamSpec(
            "max_tokens",
            "Max Tokens",
            "number",
            4096,
            minimum=256,
            maximum=8192,
            step=256,
            help="Begrenzt die Länge pro Abschnitt",
        ),
    )

    def _language_kwargs(self, language: str) -> dict[str, Any]:
        return {"lang_code": LANGUAGE_NAMES.get(language, "auto")}
