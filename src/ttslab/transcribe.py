"""Transkription der Stimmproben per mlx-whisper. Wird erst beim ersten Aufruf geladen."""

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def transcribe(audio_path: str, language: str = "de", model_id: str = DEFAULT_MODEL) -> str:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_id,
        language=None if language in ("", "auto") else language,
    )
    return result["text"].strip()
