"""Adapter-Tests für die MLX-Engines mit einem Fake-Modell, ohne Download."""

from dataclasses import dataclass

import numpy as np
import pytest

from ttslab.engines.qwen3 import Qwen3Engine
from ttslab.engines.voxcpm2 import VoxCPM2Engine
from ttslab.registry import default_registry


@dataclass
class FakeSegment:
    audio: np.ndarray
    sample_rate: int = 24000


class FakeModel:
    """Ahmt mlx-audio nach: generate() ist ein Generator über Segmente."""

    def __init__(self, segments=2):
        self.segments = segments
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        for _ in range(self.segments):
            yield FakeSegment(np.ones(1200, dtype="float32"))


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def qwen(fake_model, monkeypatch):
    monkeypatch.setattr("mlx.core.random.seed", lambda seed: None)
    return Qwen3Engine(loader=lambda model_id: fake_model)


def test_segments_are_concatenated_into_one_mono_signal(qwen, fake_model):
    result = qwen.generate("Hallo.", None, None, "de", seed=7)

    assert result.audio.shape == (2400,)
    assert result.audio.ndim == 1
    assert result.sample_rate == 24000
    assert result.seconds > 0


def test_model_is_loaded_once_and_only_on_generate(monkeypatch, fake_model):
    monkeypatch.setattr("mlx.core.random.seed", lambda seed: None)
    loads = []
    engine = Qwen3Engine(loader=lambda model_id: loads.append(model_id) or fake_model)

    assert loads == []

    engine.generate("Eins.", None, None, "de", seed=1)
    engine.generate("Zwei.", None, None, "de", seed=2)

    assert loads == ["mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"]


def test_qwen_translates_language_code_to_name(qwen, fake_model):
    qwen.generate("Hallo.", None, None, "de", seed=1)
    qwen.generate("Hello.", None, None, "kl-ingonisch", seed=1)

    assert fake_model.calls[0]["lang_code"] == "german"
    assert fake_model.calls[1]["lang_code"] == "auto"


def test_reference_audio_and_transcript_are_passed_through(qwen, fake_model):
    qwen.generate("Hallo.", "/tmp/ref.wav", "Das ist die Probe.", "de", seed=1, temperature=0.7)

    call = fake_model.calls[0]
    assert call["ref_audio"] == "/tmp/ref.wav"
    assert call["ref_text"] == "Das ist die Probe."
    assert call["temperature"] == 0.7


def test_transcript_without_audio_is_not_sent(qwen, fake_model):
    qwen.generate("Hallo.", None, "Transkript ohne Audio", "de", seed=1)

    assert "ref_audio" not in fake_model.calls[0]
    assert "ref_text" not in fake_model.calls[0]


def test_voxcpm_sends_no_language_parameter(monkeypatch, fake_model):
    monkeypatch.setattr("mlx.core.random.seed", lambda seed: None)
    engine = VoxCPM2Engine(loader=lambda model_id: fake_model)

    engine.generate("Hallo.", None, None, "de", seed=1, cfg_value=2.5)

    assert "lang_code" not in fake_model.calls[0]
    assert fake_model.calls[0]["cfg_value"] == 2.5


def test_empty_result_is_reported(monkeypatch):
    monkeypatch.setattr("mlx.core.random.seed", lambda seed: None)
    engine = Qwen3Engine(loader=lambda model_id: FakeModel(segments=0))

    with pytest.raises(RuntimeError, match="kein Audio"):
        engine.generate("Hallo.", None, None, "de", seed=1)


def test_default_registry_lists_all_three_engines():
    registry = default_registry()

    assert registry.ids == ["qwen3", "voxcpm2", "chatterbox"]
    assert registry.get("qwen3").model_id.startswith("mlx-community/")
