import numpy as np
import pytest
import soundfile as sf

from ttslab.engines.base import GenerationResult, ParamSpec
from ttslab.registry import EngineEntry, Registry
from ttslab.runs import RunStore
from ttslab.voices import VoiceStore


class FakeEngine:
    """Engine ohne Modell: erzeugt einen Sinus-Ton und merkt sich die Aufrufe."""

    params = (
        ParamSpec("temperature", "Temperatur", "slider", 0.8, minimum=0.1, maximum=1.5),
        ParamSpec("takes_note", "Notiz", "text", ""),
    )

    def __init__(self, engine_id="fake", label="Fake", needs_ref_text=False):
        self.id = engine_id
        self.label = label
        self.model_id = f"fake/{engine_id}"
        self.needs_ref_text = needs_ref_text
        self.loaded = False
        self.calls: list[dict] = []

    def load(self):
        self.loaded = True

    def generate(self, text, ref_audio, ref_text, language, seed, **params):
        self.load()
        self.calls.append(
            {
                "text": text,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "language": language,
                "seed": seed,
                **params,
            }
        )
        t = np.linspace(0, 0.5, 12000, dtype="float32")
        return GenerationResult(24000, np.sin(2 * np.pi * 220 * t), seconds=0.1)


@pytest.fixture
def fake_engines():
    return [FakeEngine("fake"), FakeEngine("fake2", label="Fake 2")]


@pytest.fixture
def registry(fake_engines):
    return Registry(
        [
            EngineEntry(engine.id, engine.label, lambda engine=engine: engine)
            for engine in fake_engines
        ]
    )


@pytest.fixture
def voice_store(tmp_path):
    store = VoiceStore(tmp_path / "voices")
    wav = tmp_path / "probe.wav"
    sf.write(wav, np.zeros(24000, dtype="float32"), 24000)
    store.add(wav, name="Bene", transcript="Das ist meine Stimmprobe.")
    return store


@pytest.fixture
def run_store(tmp_path):
    return RunStore(tmp_path / "outputs")
