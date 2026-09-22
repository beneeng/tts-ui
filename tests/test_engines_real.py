"""Smoke-Tests gegen die echten Modelle. Laufen nur mit `pytest -m slow`.

Beim ersten Lauf werden mehrere Gigabyte Modellgewichte heruntergeladen.
"""

import numpy as np
import pytest
import soundfile as sf

from ttslab.registry import default_registry

pytestmark = pytest.mark.slow

TEXT = "Guten Tag, dies ist ein kurzer Test für die deutsche Aussprache."


@pytest.fixture(scope="module")
def registry():
    return default_registry()


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Eine synthetische Stimmprobe, damit der Test ohne echte Aufnahme läuft."""
    path = tmp_path_factory.mktemp("ref") / "ref.wav"
    t = np.linspace(0, 4.0, 4 * 24000, dtype="float32")
    wave = 0.2 * np.sin(2 * np.pi * 130 * t) + 0.05 * np.sin(2 * np.pi * 260 * t)
    sf.write(path, wave, 24000)
    return str(path)


@pytest.mark.parametrize("engine_id", ["qwen3", "voxcpm2", "chatterbox"])
def test_engine_generates_mono_audio(registry, engine_id):
    engine = registry.get(engine_id)
    if engine.needs_ref_audio:
        pytest.skip(f"{engine.label} hat keine eingebaute Stimme")

    result = engine.generate(TEXT, None, None, "de", seed=1234)

    assert result.audio.ndim == 1
    assert result.sample_rate >= 16000
    assert len(result.audio) > result.sample_rate, "mindestens eine Sekunde Audio"
    assert np.isfinite(result.audio).all()


@pytest.mark.parametrize("engine_id", ["qwen3", "voxcpm2", "chatterbox"])
def test_engine_accepts_reference_audio(registry, engine_id, reference):
    engine = registry.get(engine_id)

    result = engine.generate(TEXT, reference, "Ein gleichmäßiger Testton.", "de", seed=1234)

    assert result.audio.ndim == 1
    assert len(result.audio) > result.sample_rate


def test_same_seed_gives_same_audio(registry):
    engine = registry.get("qwen3")

    first = engine.generate("Kurzer Satz.", None, None, "de", seed=7)
    second = engine.generate("Kurzer Satz.", None, None, "de", seed=7)

    assert np.allclose(first.audio, second.audio)
