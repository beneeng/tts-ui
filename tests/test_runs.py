import zipfile

import numpy as np
import pytest
import soundfile as sf

from ttslab.runs import RunStore


def add_run(store, *, engine_id="fake", voice_slug="bene", text="Hallo", seed=1):
    return store.add(
        audio=np.zeros(12000, dtype="float32"),
        sample_rate=24000,
        engine_id=engine_id,
        engine_label=engine_id.title(),
        model_id=f"{engine_id}/model",
        text=text,
        voice_slug=voice_slug,
        voice_name=voice_slug or "ohne Stimmprobe",
        language="de",
        seed=seed,
        params={"temperature": 0.8},
        generate_seconds=1.234,
    )


def test_add_writes_audio_and_metadata(tmp_path):
    store = RunStore(tmp_path)

    run = add_run(store)

    assert run.seconds == 0.5
    assert run.generate_seconds == 1.23
    assert run.params == {"temperature": 0.8}
    assert sf.info(run.audio_path).samplerate == 24000
    assert store.get(run.id) == run


def test_list_is_newest_first(tmp_path):
    store = RunStore(tmp_path)

    first = add_run(store, text="erster")
    second = add_run(store, text="zweiter")

    assert [run.id for run in store.list()] == [second.id, first.id]


def test_list_filters_by_engine_and_voice(tmp_path):
    store = RunStore(tmp_path)
    qwen = add_run(store, engine_id="qwen3", voice_slug="bene")
    add_run(store, engine_id="chatterbox", voice_slug="bene")
    add_run(store, engine_id="qwen3", voice_slug="andere")

    assert [run.id for run in store.list(engine_id="qwen3", voice_slug="bene")] == [qwen.id]
    assert len(store.list(engine_id="qwen3")) == 2


def test_favorite_survives_reload(tmp_path):
    store = RunStore(tmp_path)
    run = add_run(store)

    store.set_favorite(run.id, True)

    assert RunStore(tmp_path).get(run.id).favorite is True


def test_delete_removes_audio_and_metadata(tmp_path):
    store = RunStore(tmp_path)
    run = add_run(store)

    store.delete(run.id)

    assert store.list() == []
    with pytest.raises(KeyError):
        store.get(run.id)


def test_export_zip_contains_audio_and_metadata(tmp_path):
    store = RunStore(tmp_path / "outputs")
    run = add_run(store)

    archive_path = store.export_zip([run.id], tmp_path / "export.zip")

    with zipfile.ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [f"{run.id}.json", f"{run.id}.wav"]
