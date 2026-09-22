import pytest

from ttslab.engines.base import Engine, ParamSpec, clean_params, defaults
from ttslab.registry import EngineEntry, Registry


def test_get_is_lazy_and_cached():
    created = []

    def factory():
        from tests.conftest import FakeEngine

        engine = FakeEngine()
        created.append(engine)
        return engine

    registry = Registry([EngineEntry("fake", "Fake", factory)])

    assert registry.is_loaded("fake") is False
    assert created == []

    engine = registry.get("fake")

    assert registry.get("fake") is engine
    assert len(created) == 1
    assert engine.loaded is False, "Das Modell darf erst beim Generieren geladen werden"


def test_unknown_engine_raises(registry):
    with pytest.raises(KeyError):
        registry.get("gibt-es-nicht")


def test_choices_are_label_value_pairs(registry):
    assert registry.choices() == [("Fake", "fake"), ("Fake 2", "fake2")]


def test_fake_engine_satisfies_protocol(fake_engines):
    assert isinstance(fake_engines[0], Engine)


def test_clean_params_drops_unknown_and_casts():
    specs = (
        ParamSpec("temperature", "Temperatur", "slider", 0.8),
        ParamSpec("max_tokens", "Max Tokens", "number", 1000),
        ParamSpec("voice", "Stimme", "dropdown", "Ryan"),
    )

    cleaned = clean_params(
        specs,
        {"temperature": 1, "max_tokens": 500.0, "voice": "Aiden", "unbekannt": 3, "leer": None},
    )

    assert cleaned == {"temperature": 1.0, "max_tokens": 500, "voice": "Aiden"}
    assert isinstance(cleaned["temperature"], float)
    assert isinstance(cleaned["max_tokens"], int)


def test_defaults_come_from_specs():
    specs = (ParamSpec("temperature", "Temperatur", "slider", 0.8),)

    assert defaults(specs) == {"temperature": 0.8}
