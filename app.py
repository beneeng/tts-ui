"""TTS-Lab: lokale Web-UI zum Vergleichen von Voice-Cloning-Modellen."""

import os
from pathlib import Path

import gradio as gr

from ttslab.lab import generate_runs
from ttslab.registry import default_registry
from ttslab.runs import RunStore
from ttslab.transcribe import transcribe
from ttslab.ui import (
    LANGUAGES,
    NO_VOICE,
    RUN_COLUMNS,
    VOICE_COLUMNS,
    group_params,
    run_label,
    run_rows,
    voice_choices,
    voice_rows,
    voice_slug_or_none,
)
from ttslab.voices import VoiceStore

DATA_DIR = Path(os.environ.get("TTSLAB_DATA", "data"))
VOICES = VoiceStore(DATA_DIR / "voices")
RUNS = RunStore(DATA_DIR / "outputs")
REGISTRY = default_registry()

MAX_TAKES = 5
MAX_RESULTS = len(REGISTRY.ids) * MAX_TAKES
MAX_COMPARE = 6

# Zuordnung Regler -> (engine_id, parameter). Wird beim Aufbau der UI gefüllt.
PARAM_KEYS: list[tuple[str, str]] = []


def run_checkbox_choices(runs) -> list[tuple[str, str]]:
    return [
        (
            f"{run.created_at[11:19]} · {run.engine_label} · {run.voice_name} · Seed {run.seed}",
            run.id,
        )
        for run in runs
    ]


# --- Tab: Generieren -------------------------------------------------------


def on_generate(
    text,
    language,
    voice_choice,
    engine_ids,
    takes,
    seed,
    *param_values,
    progress=gr.Progress(),
):
    params_by_engine = group_params(PARAM_KEYS, list(param_values))
    base_seed = int(seed) if seed not in (None, "") else None
    takes = int(takes)
    expected = max(len(engine_ids) * takes, 1)
    produced = []

    try:
        runs_iter = generate_runs(
            registry=REGISTRY,
            voice_store=VOICES,
            run_store=RUNS,
            text=text,
            engine_ids=list(engine_ids),
            params_by_engine=params_by_engine,
            voice_slug=voice_slug_or_none(voice_choice),
            language=language,
            takes=takes,
            base_seed=base_seed,
        )
        for index, run in enumerate(runs_iter, start=1):
            produced.append(run)
            progress(index / expected, desc=f"{index}/{expected}: {run.engine_label}")
            yield result_updates(produced, f"{index} von {expected} erzeugt …")
    except ValueError as error:
        raise gr.Error(str(error)) from error
    except Exception as error:  # Modellfehler sollen die UI nicht abwerfen
        raise gr.Error(f"Fehler beim Generieren: {error}") from error

    yield result_updates(produced, f"Fertig: {len(produced)} Ergebnisse.")


def result_updates(runs, status):
    """Die Ergebnis-Slots füllen und den Vergleichs-Tab nachziehen."""
    updates = []
    for index in range(MAX_RESULTS):
        if index < len(runs):
            run = runs[index]
            updates.append(gr.update(value=run_label(run), visible=True))
            updates.append(gr.update(value=run.audio_path, visible=True, label=run.engine_label))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=None, visible=False))

    all_runs = RUNS.list()
    return [
        status,
        *updates,
        gr.update(value=run_rows(all_runs)),
        gr.update(choices=run_checkbox_choices(all_runs)),
    ]


# --- Tab: Stimmproben ------------------------------------------------------


def on_save_voice(audio_path, name, language, note, auto_transcribe):
    if not audio_path:
        raise gr.Error("Bitte erst aufnehmen oder eine Datei hochladen.")
    if not name.strip():
        raise gr.Error("Bitte einen Namen für die Stimmprobe vergeben.")

    sample = VOICES.add(audio_path, name=name, language=language, note=note)
    status = f"Stimmprobe „{sample.name}“ gespeichert."
    if auto_transcribe:
        try:
            sample = VOICES.update(
                sample.slug, transcript=transcribe(sample.audio_path, language=language)
            )
            status += " Transkript erstellt."
        except Exception as error:  # Whisper ist optional, die Probe bleibt trotzdem
            status += f" Transkription fehlgeschlagen: {error}"

    samples = VOICES.list()
    return (
        status,
        gr.update(value=voice_rows(samples)),
        gr.update(choices=voice_choices(samples), value=sample.slug),
        gr.update(choices=[("alle Stimmen", "alle"), *voice_choices(samples)[1:]]),
        sample.slug,
        sample.name,
        sample.transcript,
        sample.note,
        sample.audio_path,
    )


def on_pick_voice(event: gr.SelectData):
    sample = VOICES.list()[event.index[0]]
    return sample.slug, sample.name, sample.transcript, sample.note, sample.audio_path


def on_update_voice(slug, name, transcript, note):
    if not slug:
        raise gr.Error("Bitte erst eine Stimmprobe in der Liste auswählen.")
    VOICES.update(slug, name=name, transcript=transcript, note=note)
    samples = VOICES.list()
    return (
        f"„{name}“ aktualisiert.",
        gr.update(value=voice_rows(samples)),
        gr.update(choices=voice_choices(samples)),
        gr.update(choices=[("alle Stimmen", "alle"), *voice_choices(samples)[1:]]),
    )


def on_delete_voice(slug):
    if not slug:
        raise gr.Error("Bitte erst eine Stimmprobe in der Liste auswählen.")
    VOICES.delete(slug)
    samples = VOICES.list()
    return (
        "Stimmprobe gelöscht.",
        gr.update(value=voice_rows(samples)),
        gr.update(choices=voice_choices(samples), value=NO_VOICE),
        gr.update(choices=[("alle Stimmen", "alle"), *voice_choices(samples)[1:]], value="alle"),
        "",
        "",
        "",
        "",
        None,
    )


def on_retranscribe(slug, language):
    if not slug:
        raise gr.Error("Bitte erst eine Stimmprobe in der Liste auswählen.")
    sample = VOICES.get(slug)
    text = transcribe(sample.audio_path, language=language or sample.language)
    VOICES.update(slug, transcript=text)
    return "Transkript neu erstellt.", text, gr.update(value=voice_rows(VOICES.list()))


# --- Tab: Vergleich --------------------------------------------------------


def filtered_runs(engine_filter, voice_filter):
    return RUNS.list(
        engine_id=None if engine_filter in (None, "", "alle") else engine_filter,
        voice_slug=None if voice_filter in (None, "", "alle") else voice_filter,
    )


def on_refresh_runs(engine_filter, voice_filter):
    runs = filtered_runs(engine_filter, voice_filter)
    return gr.update(value=run_rows(runs)), gr.update(choices=run_checkbox_choices(runs))


def on_compare(selected_ids):
    updates = []
    for index in range(MAX_COMPARE):
        if index < len(selected_ids):
            run = RUNS.get(selected_ids[index])
            updates.append(gr.update(value=run_label(run), visible=True))
            updates.append(gr.update(value=run.audio_path, visible=True, label=run.engine_label))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=None, visible=False))

    hint = ""
    if len(selected_ids) > MAX_COMPARE:
        hint = f"Es werden nur die ersten {MAX_COMPARE} Auswahlen angezeigt."
    return [hint, *updates]


def on_toggle_favorite(selected_ids, engine_filter, voice_filter):
    if not selected_ids:
        raise gr.Error("Bitte erst Ergebnisse auswählen.")
    for run_id in selected_ids:
        RUNS.set_favorite(run_id, not RUNS.get(run_id).favorite)
    table, picker = on_refresh_runs(engine_filter, voice_filter)
    return "Favoriten umgeschaltet.", table, picker


def on_delete_runs(selected_ids, engine_filter, voice_filter):
    if not selected_ids:
        raise gr.Error("Bitte erst Ergebnisse auswählen.")
    for run_id in selected_ids:
        RUNS.delete(run_id)
    table, picker = on_refresh_runs(engine_filter, voice_filter)
    picker["value"] = []  # sonst zeigt die Auswahl auf gelöschte Läufe
    return f"{len(selected_ids)} Ergebnisse gelöscht.", table, picker


def on_export_runs(selected_ids):
    if not selected_ids:
        raise gr.Error("Bitte erst Ergebnisse auswählen.")
    return str(RUNS.export_zip(selected_ids, DATA_DIR / "export.zip"))


# --- Oberfläche ------------------------------------------------------------


def param_component(spec):
    common = {"label": spec.label, "value": spec.default, "info": spec.help}
    if spec.kind == "slider":
        return gr.Slider(minimum=spec.minimum, maximum=spec.maximum, step=spec.step, **common)
    if spec.kind == "number":
        return gr.Number(minimum=spec.minimum, maximum=spec.maximum, step=spec.step, **common)
    if spec.kind == "checkbox":
        return gr.Checkbox(**common)
    if spec.kind == "dropdown":
        return gr.Dropdown(choices=list(spec.choices or ()), **common)
    return gr.Textbox(**common)


def build_param_components():
    """Pro Engine ein Accordion mit ihren Reglern. Füllt PARAM_KEYS."""
    accordions, components = [], []
    for engine_id in REGISTRY.ids:
        engine = REGISTRY.get(engine_id)
        with gr.Accordion(f"Parameter: {engine.label}", open=False) as accordion:
            hint = " · braucht eine Stimmprobe" if engine.needs_ref_audio else ""
            gr.Markdown(f"Modell: `{engine.model_id}`{hint}")
            for spec in engine.params:
                components.append(param_component(spec))
                PARAM_KEYS.append((engine_id, spec.name))
        accordions.append(accordion)
    return accordions, components


with gr.Blocks(title="TTS-Lab") as demo:
    gr.Markdown("# 🎙️ TTS-Lab\nStimmproben verwalten, Modelle vergleichen, Ergebnisse aufheben.")

    samples_at_start = VOICES.list()
    runs_at_start = RUNS.list()

    with gr.Tabs():
        # ---------------- Generieren ----------------
        with gr.Tab("Generieren"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Text",
                        lines=6,
                        placeholder="Der Text, der vorgelesen werden soll …",
                    )
                    with gr.Row():
                        language = gr.Dropdown(
                            choices=list(LANGUAGES), value="de", label="Sprache"
                        )
                        voice_pick = gr.Dropdown(
                            choices=voice_choices(samples_at_start),
                            value=NO_VOICE,
                            label="Stimmprobe",
                        )
                    engines_pick = gr.CheckboxGroup(
                        choices=REGISTRY.choices(),
                        value=REGISTRY.ids,
                        label="Engines (jede erzeugt ein eigenes Ergebnis)",
                    )
                    with gr.Row():
                        takes = gr.Slider(
                            minimum=1,
                            maximum=MAX_TAKES,
                            step=1,
                            value=1,
                            label="Takes je Engine",
                        )
                        seed_input = gr.Number(
                            label="Seed (leer = zufällig)", value=None, precision=0
                        )
                    generate_btn = gr.Button("🔊 Generieren", variant="primary")
                    status = gr.Markdown()

                with gr.Column(scale=2):
                    param_accordions, param_components = build_param_components()

            gr.Markdown("### Ergebnisse dieses Durchlaufs")
            result_labels, result_players = [], []
            for _ in range(MAX_RESULTS):
                result_labels.append(gr.Markdown(visible=False))
                result_players.append(gr.Audio(visible=False, type="filepath"))

        # ---------------- Stimmproben ----------------
        with gr.Tab("Stimmproben"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Neue Stimmprobe")
                    new_audio = gr.Audio(
                        sources=["microphone", "upload"], type="filepath", label="Aufnahme"
                    )
                    new_name = gr.Textbox(label="Name", placeholder="z. B. Bene neutral")
                    new_language = gr.Dropdown(
                        choices=list(LANGUAGES), value="de", label="Sprache der Aufnahme"
                    )
                    new_note = gr.Textbox(label="Notiz", placeholder="Mikrofon, Raum, Datum …")
                    auto_transcribe = gr.Checkbox(
                        value=True, label="Transkript automatisch erstellen (Whisper, lokal)"
                    )
                    save_voice_btn = gr.Button("Speichern", variant="primary")
                    voice_status = gr.Markdown()

                with gr.Column():
                    gr.Markdown("### Vorhandene Stimmproben")
                    voice_table = gr.Dataframe(
                        headers=list(VOICE_COLUMNS),
                        value=voice_rows(samples_at_start),
                        interactive=False,
                        label="Zum Bearbeiten Zeile anklicken",
                    )
                    selected_slug = gr.Textbox(visible=False)
                    detail_name = gr.Textbox(label="Name")
                    detail_transcript = gr.Textbox(
                        label="Transkript",
                        lines=3,
                        info="Engines mit Transkript-Unterstützung klonen damit genauer",
                    )
                    detail_note = gr.Textbox(label="Notiz")
                    detail_player = gr.Audio(
                        label="Anhören und herunterladen", type="filepath"
                    )
                    with gr.Row():
                        update_voice_btn = gr.Button("Änderungen speichern")
                        retranscribe_btn = gr.Button("Transkript neu erstellen")
                        delete_voice_btn = gr.Button("Löschen", variant="stop")

        # ---------------- Vergleich ----------------
        with gr.Tab("Vergleich"):
            with gr.Row():
                engine_filter = gr.Dropdown(
                    choices=[("alle Engines", "alle"), *REGISTRY.choices()],
                    value="alle",
                    label="Filter Engine",
                )
                voice_filter = gr.Dropdown(
                    choices=[("alle Stimmen", "alle"), *voice_choices(samples_at_start)[1:]],
                    value="alle",
                    label="Filter Stimmprobe",
                )
                refresh_btn = gr.Button("Aktualisieren")

            runs_table = gr.Dataframe(
                headers=list(RUN_COLUMNS),
                value=run_rows(runs_at_start),
                interactive=False,
                label="Alle Ergebnisse",
            )
            runs_pick = gr.CheckboxGroup(
                choices=run_checkbox_choices(runs_at_start),
                label=f"Zum Vergleich auswählen (max. {MAX_COMPARE} Player)",
            )
            with gr.Row():
                compare_btn = gr.Button("Auswahl anhören", variant="primary")
                favorite_btn = gr.Button("★ umschalten")
                export_btn = gr.Button("Als ZIP exportieren")
                delete_runs_btn = gr.Button("Löschen", variant="stop")
            compare_status = gr.Markdown()
            export_file = gr.File(label="Export")

            compare_labels, compare_players = [], []
            for _ in range(MAX_COMPARE):
                compare_labels.append(gr.Markdown(visible=False))
                compare_players.append(gr.Audio(visible=False, type="filepath"))

    # ---------------- Verdrahtung ----------------
    result_slots = [item for pair in zip(result_labels, result_players) for item in pair]
    compare_slots = [item for pair in zip(compare_labels, compare_players) for item in pair]

    engines_pick.change(
        fn=lambda selected: [gr.update(visible=eid in selected) for eid in REGISTRY.ids],
        inputs=engines_pick,
        outputs=param_accordions,
    )

    generate_btn.click(
        fn=on_generate,
        inputs=[
            text_input,
            language,
            voice_pick,
            engines_pick,
            takes,
            seed_input,
            *param_components,
        ],
        outputs=[status, *result_slots, runs_table, runs_pick],
    )

    voice_outputs = [
        voice_status,
        voice_table,
        voice_pick,
        voice_filter,
        selected_slug,
        detail_name,
        detail_transcript,
        detail_note,
        detail_player,
    ]
    save_voice_btn.click(
        fn=on_save_voice,
        inputs=[new_audio, new_name, new_language, new_note, auto_transcribe],
        outputs=voice_outputs,
    )
    voice_table.select(
        fn=on_pick_voice,
        outputs=[selected_slug, detail_name, detail_transcript, detail_note, detail_player],
    )
    update_voice_btn.click(
        fn=on_update_voice,
        inputs=[selected_slug, detail_name, detail_transcript, detail_note],
        outputs=[voice_status, voice_table, voice_pick, voice_filter],
    )
    retranscribe_btn.click(
        fn=on_retranscribe,
        inputs=[selected_slug, new_language],
        outputs=[voice_status, detail_transcript, voice_table],
    )
    delete_voice_btn.click(fn=on_delete_voice, inputs=selected_slug, outputs=voice_outputs)

    refresh_btn.click(
        fn=on_refresh_runs,
        inputs=[engine_filter, voice_filter],
        outputs=[runs_table, runs_pick],
    )
    compare_btn.click(fn=on_compare, inputs=runs_pick, outputs=[compare_status, *compare_slots])
    favorite_btn.click(
        fn=on_toggle_favorite,
        inputs=[runs_pick, engine_filter, voice_filter],
        outputs=[compare_status, runs_table, runs_pick],
    )
    delete_runs_btn.click(
        fn=on_delete_runs,
        inputs=[runs_pick, engine_filter, voice_filter],
        outputs=[compare_status, runs_table, runs_pick],
    )
    export_btn.click(fn=on_export_runs, inputs=runs_pick, outputs=export_file)


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        allowed_paths=[str(DATA_DIR.resolve())],
    )
