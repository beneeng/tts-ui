"""Tests der UI-Callbacks mit Fake-Engines, ohne Modelle und ohne Browser."""

import importlib

import numpy as np
import pytest
import soundfile as sf

from ttslab.registry import EngineEntry, Registry
from ttslab.runs import RunStore
from ttslab.voices import VoiceStore

from .conftest import FakeEngine


@pytest.fixture
def app(tmp_path, monkeypatch, fake_engines):
    """app.py mit eigenem Datenverzeichnis und Fake-Engines laden."""
    monkeypatch.setenv("TTSLAB_DATA", str(tmp_path))
    import app as app_module

    app_module = importlib.reload(app_module)
    app_module.REGISTRY = Registry(
        [
            EngineEntry(engine.id, engine.label, lambda engine=engine: engine)
            for engine in fake_engines
        ]
    )
    app_module.VOICES = VoiceStore(tmp_path / "voices")
    app_module.RUNS = RunStore(tmp_path / "outputs")
    app_module.PARAM_KEYS = [("fake", "temperature"), ("fake2", "temperature")]
    return app_module


def generate(app, **kwargs):
    """on_generate ist ein Generator. Liefert die Updates des letzten Schritts."""
    params = dict(
        text="Hallo Welt.",
        language="de",
        voice_choice=app.NO_VOICE,
        engine_ids=["fake", "fake2"],
        takes=1,
        seed=42,
        param_values=(0.8, 0.9),
    )
    params.update(kwargs)
    steps = list(
        app.on_generate(
            params["text"],
            params["language"],
            params["voice_choice"],
            params["engine_ids"],
            params["takes"],
            params["seed"],
            *params["param_values"],
            progress=lambda *a, **k: None,
        )
    )
    return steps


def visible_players(app, updates):
    """Die Audio-Slots aus dem Update-Paket zählen, die sichtbar geschaltet wurden."""
    slots = updates[1 : 1 + app.MAX_RESULTS * 2]
    players = slots[1::2]
    return [update for update in players if update.get("visible")]


def test_generate_fills_one_player_per_engine_and_take(app, fake_engines):
    steps = generate(app, takes=2)

    last = steps[-1]
    assert last[0] == "Fertig: 4 Ergebnisse."
    assert len(visible_players(app, last)) == 4
    assert len(app.RUNS.list()) == 4
    assert len(steps) == 5, "Nach jedem Ergebnis ein Zwischenstand plus Endstand"


def test_generate_passes_parameters_per_engine(app, fake_engines):
    generate(app, param_values=(0.3, 1.2))

    assert fake_engines[0].calls[0]["temperature"] == 0.3
    assert fake_engines[1].calls[0]["temperature"] == 1.2


def test_generate_uses_the_chosen_voice_sample(app, tmp_path, fake_engines):
    wav = tmp_path / "probe.wav"
    sf.write(wav, np.zeros(24000, dtype="float32"), 24000)
    app.VOICES.add(wav, name="Bene", transcript="Probe.")

    generate(app, voice_choice="bene")

    assert fake_engines[0].calls[0]["ref_audio"].endswith("voices/bene/audio.wav")
    assert app.RUNS.list()[0].voice_name == "Bene"


def test_generate_without_text_raises_ui_error(app):
    import gradio as gr

    with pytest.raises(gr.Error):
        generate(app, text="   ")


def test_unused_result_slots_are_hidden(app):
    last = generate(app, engine_ids=["fake"], takes=1)[-1]

    players = last[1 : 1 + app.MAX_RESULTS * 2][1::2]
    assert sum(1 for update in players if update.get("visible")) == 1
    assert sum(1 for update in players if update.get("visible") is False) == app.MAX_RESULTS - 1


def test_save_voice_without_transcription(app, tmp_path):
    wav = tmp_path / "neu.wav"
    sf.write(wav, np.zeros(24000, dtype="float32"), 24000)

    status, *_ = app.on_save_voice(str(wav), "Bene", "de", "Notiz", False)

    assert "gespeichert" in status
    assert [sample.name for sample in app.VOICES.list()] == ["Bene"]


def test_save_voice_survives_failing_transcription(app, tmp_path, monkeypatch):
    wav = tmp_path / "neu.wav"
    sf.write(wav, np.zeros(24000, dtype="float32"), 24000)
    monkeypatch.setattr(
        app, "transcribe", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kein Modell"))
    )

    status, *_ = app.on_save_voice(str(wav), "Bene", "de", "", True)

    assert "fehlgeschlagen" in status
    assert len(app.VOICES.list()) == 1


def test_compare_shows_selected_runs_and_warns_above_limit(app):
    generate(app, takes=2)
    run_ids = [run.id for run in app.RUNS.list()]

    hint, *slots = app.on_compare(run_ids[:2])

    assert hint == ""
    assert sum(1 for update in slots[1::2] if update.get("visible")) == 2

    hint_many, *_ = app.on_compare(run_ids * 4)
    assert "ersten 6" in hint_many


def test_toggle_favorite_and_delete_runs(app):
    generate(app, engine_ids=["fake"])
    run_id = app.RUNS.list()[0].id

    app.on_toggle_favorite([run_id], "alle", "alle")
    assert app.RUNS.get(run_id).favorite is True

    status, _table, picker = app.on_delete_runs([run_id], "alle", "alle")
    assert status.startswith("1 Ergebnisse")
    assert app.RUNS.list() == []
    assert picker["value"] == [], "die Auswahl darf nicht auf gelöschte Läufe zeigen"


def test_actions_without_selection_raise(app):
    import gradio as gr

    for callback in (app.on_toggle_favorite, app.on_delete_runs):
        with pytest.raises(gr.Error):
            callback([], "alle", "alle")
    with pytest.raises(gr.Error):
        app.on_export_runs([])


def test_run_filter_by_engine(app):
    generate(app)

    assert len(app.filtered_runs("fake", "alle")) == 1
    assert len(app.filtered_runs("alle", "alle")) == 2


def test_param_keys_cover_every_engine_parameter():
    """Die echte UI muss für jeden Parameter jeder Engine einen Regler anlegen."""
    import importlib

    import app as app_module

    app_module = importlib.reload(app_module)
    expected = [
        (engine_id, spec.name)
        for engine_id in app_module.REGISTRY.ids
        for spec in app_module.REGISTRY.get(engine_id).params
    ]

    assert app_module.PARAM_KEYS == expected
    assert len(app_module.PARAM_KEYS) > 10


def test_engine_error_becomes_ui_error(app, fake_engines, monkeypatch):
    import gradio as gr

    def boom(*args, **kwargs):
        raise RuntimeError("Modell kaputt")

    monkeypatch.setattr(fake_engines[0], "generate", boom)

    with pytest.raises(gr.Error, match="Modell kaputt"):
        generate(app, engine_ids=["fake"])


def test_generate_returns_exactly_the_wired_outputs(app):
    """Gradio bricht zur Laufzeit ab, wenn Anzahl der Rückgaben und Outputs auseinanderlaufen."""
    last = generate(app)[-1]

    wired = 1 + len(app.result_slots) + 2  # Status + Ergebnis-Slots + Tabelle + Auswahl
    assert len(last) == wired


def test_compare_returns_exactly_the_wired_outputs(app):
    generate(app)
    run_id = app.RUNS.list()[0].id

    assert len(app.on_compare([run_id])) == 1 + len(app.compare_slots)
