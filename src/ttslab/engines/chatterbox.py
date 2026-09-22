"""Chatterbox Multilingual (Resemble AI) über mlx-audio, als Vergleichsbasis."""

from typing import Any

from .base import ParamSpec
from .mlx_engine import MlxEngine

MODEL_ID = "mlx-community/chatterbox-multilingual-v3"


class ChatterboxEngine(MlxEngine):
    id = "chatterbox"
    label = "Chatterbox Multilingual"
    model_id = MODEL_ID
    needs_ref_text = False
    # Das mehrsprachige MLX-Repo enthält keine conds.safetensors, also keine Standardstimme.
    needs_ref_audio = True
    params = (
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
            help="0 empfiehlt Resemble AI, wenn die Stimmprobe eine andere Sprache spricht",
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

    def _language_kwargs(self, language: str) -> dict[str, Any]:
        return {"lang_code": language}
