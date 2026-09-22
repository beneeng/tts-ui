"""VoxCPM2 (OpenBMB, Apache 2.0) über mlx-audio.

Kennt keinen Sprachparameter, die Sprache ergibt sich aus dem Text. Das Transkript der
Stimmprobe verbessert laut Projekt die Ähnlichkeit deutlich, deshalb needs_ref_text.
"""

from .base import ParamSpec
from .mlx_engine import MlxEngine

MODEL_ID = "mlx-community/VoxCPM2-bf16"


class VoxCPM2Engine(MlxEngine):
    id = "voxcpm2"
    label = "VoxCPM2"
    model_id = MODEL_ID
    needs_ref_text = False
    params = (
        ParamSpec(
            "cfg_value",
            "CFG-Wert",
            "slider",
            2.0,
            minimum=1.0,
            maximum=5.0,
            step=0.1,
            help="Höher = näher an Text und Stimmprobe, niedriger = freier",
        ),
        ParamSpec(
            "inference_timesteps",
            "Diffusionsschritte",
            "slider",
            10,
            minimum=4,
            maximum=50,
            step=1,
            help="Mehr Schritte = sauberer, aber langsamer",
        ),
        ParamSpec("max_tokens", "Max Tokens", "number", 4096, minimum=256, maximum=8192, step=256),
    )
