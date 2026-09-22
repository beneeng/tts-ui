import numpy as np
import pytest
import soundfile as sf

from ttslab.voices import VoiceStore, slugify


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "probe.wav"
    sf.write(path, np.zeros(24000, dtype="float32"), 24000)
    return path


def test_slugify_handles_umlauts_and_spaces():
    assert slugify("Meine Stimme Nr. 2") == "meine-stimme-nr-2"
    assert slugify("!!!") == "stimme"


def test_add_stores_audio_and_metadata(tmp_path, wav):
    store = VoiceStore(tmp_path / "voices")

    sample = store.add(wav, name="Bene", transcript="Hallo", note="Aufnahme am Laptop")

    assert sample.slug == "bene"
    assert sample.seconds == 1.0
    assert store.get("bene") == sample
    assert sf.info(sample.audio_path).samplerate == 24000


def test_add_mixes_stereo_down_to_mono(tmp_path):
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.zeros((24000, 2), dtype="float32"), 24000)
    store = VoiceStore(tmp_path / "voices")

    sample = store.add(stereo, name="Stereo")

    assert sf.info(sample.audio_path).channels == 1


def test_duplicate_names_get_distinct_slugs(tmp_path, wav):
    store = VoiceStore(tmp_path / "voices")

    first = store.add(wav, name="Bene")
    second = store.add(wav, name="Bene")

    assert (first.slug, second.slug) == ("bene", "bene-2")


def test_update_transcript_and_reject_unknown_fields(tmp_path, wav):
    store = VoiceStore(tmp_path / "voices")
    store.add(wav, name="Bene")

    updated = store.update("bene", transcript="Korrigierter Text")

    assert store.get("bene").transcript == "Korrigierter Text"
    assert updated.name == "Bene"
    with pytest.raises(ValueError):
        store.update("bene", seconds=5.0)


def test_list_is_newest_first_and_delete_removes(tmp_path, wav):
    store = VoiceStore(tmp_path / "voices")
    store.add(wav, name="Alt")
    store.add(wav, name="Neu")

    assert [s.slug for s in store.list()] == ["neu", "alt"]

    store.delete("neu")

    assert [s.slug for s in store.list()] == ["alt"]
    with pytest.raises(KeyError):
        store.get("neu")
