"""Chatterbox TTS — Lokale Web-UI mit Voice-Cloning"""

import gradio as gr
import numpy as np
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS

# Modell laden (MPS auf Apple Silicon, sonst Fallback auf CPU durch die Library)
print("🐶 Lade Chatterbox Modell …")
model = ChatterboxMultilingualTTS.from_pretrained(device="mps")
print(f"✅ Modell bereit! Sample Rate: {model.sr} Hz")


def generate_speech(
    text: str,
    ref_audio: str | None,
    language: str,
    emotion: float,
) -> tuple[int, np.ndarray]:
    """Text in Speech umwandeln."""
    if not text.strip():
        raise gr.Error("Bitte Text eingeben!")

    kwargs = {
        "text": text,
        "language_id": language,
        "temperature": 0.8,
        "top_p": 1.0,
        "cfg_weight": 0.5,
        "exaggeration": emotion,
    }

    if ref_audio:
        kwargs["audio_prompt_path"] = ref_audio

    wav = model.generate(**kwargs)
    return (model.sr, wav.squeeze(0).cpu().numpy())


# --- Gradio UI ---
with gr.Blocks(title="Chatterbox TTS") as demo:
    gr.Markdown(
        """
# 🐶 Chatterbox TTS

Open-Source Text-to-Speech mit Voice-Cloning. Läuft komplett lokal.
"""
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Text",
                placeholder="Schreibe hier, was vorgelesen werden soll …",
                lines=4,
            )
            ref_audio = gr.Audio(
                label="Stimme clonen (5–10 Sek. Audio, optional)",
                type="filepath",
            )
            language = gr.Dropdown(
                choices=[(name, code) for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda kv: kv[1])],
                value="de",
                label="Sprache",
            )
            emotion = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                label="Emotion-Exaggeration (0 = neutral, 1 = übertrieben)",
            )
            btn = gr.Button("🔊 Generieren", variant="primary")

        with gr.Column(scale=2):
            output = gr.Audio(label="Audio-Ausgabe", type="numpy")

    btn.click(
        fn=generate_speech,
        inputs=[text_input, ref_audio, language, emotion],
        outputs=output,
    )

    gr.Markdown(
        """
> 💡 **Tipp:** Lade ein kurzes Audio-File hoch, um eine Stimme zu clonen.
> Ohne Reference-File werden die Standard-Stimmen verwendet.
>
> 📖 [GitHub](https://github.com/resemble-ai/chatterbox) · MIT-Lizenz
> """
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
