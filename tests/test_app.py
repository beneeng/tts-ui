"""Smoke-Tests gegen das echte Modell (lädt beim ersten Lauf die Gewichte herunter)."""

import numpy as np
import pytest
import soundfile as sf

import app


def test_german_is_default_language():
    assert app.language.value == "de"


def test_generate_without_reference_returns_mono_audio():
    sr, wav = app.generate_speech("Guten Tag, wie geht es dir?", None, "de", 0.5)

    assert sr == app.model.sr
    assert wav.ndim == 1
    assert len(wav) > sr  # mehr als eine Sekunde Audio


def test_generate_with_reference_returns_mono_audio(tmp_path):
    sr, ref = app.generate_speech("Das ist eine Stimmprobe für den Test.", None, "de", 0.5)
    ref_path = tmp_path / "ref.wav"
    sf.write(ref_path, ref, sr)

    sr, wav = app.generate_speech("Hallo mit geklonter Stimme.", str(ref_path), "de", 0.5)

    assert wav.ndim == 1
    assert len(wav) > sr // 2


def test_empty_text_is_rejected():
    with pytest.raises(app.gr.Error):
        app.generate_speech("   ", None, "de", 0.5)
