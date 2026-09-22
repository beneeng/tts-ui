import pytest

from ttslab.lab import generate_runs, seeds_for


def run_all(**kwargs):
    return list(generate_runs(**kwargs))


def test_every_engine_and_take_produces_a_run(registry, voice_store, run_store, fake_engines):
    runs = run_all(
        registry=registry,
        voice_store=voice_store,
        run_store=run_store,
        text="Hallo Welt.",
        engine_ids=["fake", "fake2"],
        params_by_engine={"fake": {"temperature": 0.9}, "fake2": {}},
        voice_slug="bene",
        takes=2,
        base_seed=100,
    )

    assert len(runs) == 4
    assert [run.seed for run in runs] == [100, 101, 100, 101]
    assert {run.engine_id for run in runs} == {"fake", "fake2"}
    assert len(run_store.list()) == 4
    assert fake_engines[0].calls[0]["temperature"] == 0.9


def test_reference_audio_and_transcript_reach_the_engine(
    registry, voice_store, run_store, fake_engines
):
    run_all(
        registry=registry,
        voice_store=voice_store,
        run_store=run_store,
        text="Hallo.",
        engine_ids=["fake"],
        params_by_engine={},
        voice_slug="bene",
    )

    call = fake_engines[0].calls[0]
    assert call["ref_audio"] == voice_store.get("bene").audio_path
    assert call["ref_text"] == "Das ist meine Stimmprobe."


def test_without_voice_the_run_is_marked_accordingly(registry, voice_store, run_store):
    runs = run_all(
        registry=registry,
        voice_store=voice_store,
        run_store=run_store,
        text="Hallo.",
        engine_ids=["fake"],
        params_by_engine={},
        voice_slug=None,
    )

    assert runs[0].voice_slug is None
    assert runs[0].voice_name == "ohne Stimmprobe"


def test_unknown_params_never_reach_the_engine(registry, voice_store, run_store, fake_engines):
    runs = run_all(
        registry=registry,
        voice_store=voice_store,
        run_store=run_store,
        text="Hallo.",
        engine_ids=["fake"],
        params_by_engine={"fake": {"temperature": 0.7, "cfg_weight": 0.5}},
    )

    assert "cfg_weight" not in fake_engines[0].calls[0]
    assert runs[0].params == {"temperature": 0.7}


def test_engine_needing_a_transcript_rejects_missing_voice(voice_store, run_store):
    from ttslab.registry import EngineEntry, Registry

    from .conftest import FakeEngine

    engine = FakeEngine("braucht-text", needs_ref_text=True)
    registry = Registry([EngineEntry(engine.id, engine.label, lambda: engine)])

    with pytest.raises(ValueError, match="Stimmprobe"):
        run_all(
            registry=registry,
            voice_store=voice_store,
            run_store=run_store,
            text="Hallo.",
            engine_ids=["braucht-text"],
            params_by_engine={},
            voice_slug=None,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"text": "   "}, "Text"),
        ({"engine_ids": []}, "Engine"),
        ({"takes": 0}, "Take"),
    ],
)
def test_invalid_input_is_rejected(registry, voice_store, run_store, kwargs, message):
    base = dict(
        registry=registry,
        voice_store=voice_store,
        run_store=run_store,
        text="Hallo.",
        engine_ids=["fake"],
        params_by_engine={},
    )

    with pytest.raises(ValueError, match=message):
        run_all(**{**base, **kwargs})


def test_seeds_are_random_but_distinct_without_base_seed():
    seeds = seeds_for(3, None)

    assert len(set(seeds)) == 3
    assert seeds_for(3, 5) == [5, 6, 7]
